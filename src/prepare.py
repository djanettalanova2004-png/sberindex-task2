"""Панель расходов: цель, доли категорий, соседи, макро и новости."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_MAP = {
    "все категории": "total",
    "продовольствие": "food",
    "здоровье": "health",
    "общественное питание": "cafe",
    "транспорт": "transport",
    "маркетплейсы": "market",
}


def month_start(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.slice(0, 7) + "-01"
    return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")


def working_day_table() -> pd.DataFrame:
    import holidays

    try:
        ru = holidays.country_holidays("RU", years=[2023, 2024])
    except Exception:
        ru = holidays.RU(years=[2023, 2024])
    rows = []
    for start in pd.date_range("2023-01-01", "2024-12-01", freq="MS"):
        days = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")
        n = sum(1 for day in days if day.weekday() < 5 and day.date() not in ru)
        rows.append({"date": start, "n_working_days": int(n)})
    return pd.DataFrame(rows)


def _norm(text: str) -> str:
    return str(text).strip().lower().replace("ё", "е")


def _rename_categories(frame: pd.DataFrame) -> pd.DataFrame:
    names = {_norm(value): value for value in frame["category"].unique()}
    rename = {}
    for raw_name, column in CATEGORY_MAP.items():
        key = _norm(raw_name)
        if key not in names:
            raise KeyError(f"в данных нет категории «{raw_name}». Есть: {list(names)}")
        rename[names[key]] = column
    out = frame.copy()
    out["category"] = out["category"].map(rename)
    return out


def _neighbor_means(raw: Path, panel: pd.DataFrame, k: int, max_km: float) -> pd.DataFrame:
    edges = pd.read_parquet(raw / "connection.parquet", columns=["territory_id_x", "territory_id_y", "distance", "type"])
    ids = set(panel["territory_id"].unique())
    edges = edges.loc[edges["type"].astype(str) == "highway"]
    edges = edges.loc[
        edges["territory_id_x"].isin(ids)
        & edges["territory_id_y"].isin(ids)
        & (edges["territory_id_x"] != edges["territory_id_y"])
        & (edges["distance"] > 0)
        & (edges["distance"] <= max_km)
    ]
    edges = edges.sort_values(["territory_id_x", "distance"])
    edges = edges.groupby("territory_id_x", sort=False).head(k)
    spend = panel[["territory_id", "date", "total"]]
    merged = edges.merge(spend, left_on="territory_id_y", right_on="territory_id", how="inner")
    neigh = (
        merged.groupby(["territory_id_x", "date"], as_index=False)["total"]
        .mean()
        .rename(columns={"territory_id_x": "territory_id", "total": "neighbor_mean"})
    )
    return neigh


def region_mask(region: pd.Series, key: str) -> pd.Series:
    """Субъект по ключу. «Москва» не захватывает Московскую область."""
    text = region.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False)
    needle = str(key).lower().replace("ё", "е")
    if needle == "москва":
        return text.str.contains("москва", regex=False) & ~text.str.contains("московск", regex=False)
    return text.str.contains(needle, regex=False)


def _news_features(dictionary: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(ROOT / "data_static" / "events.csv", sep=";")
    events["date"] = month_start(events["month"])
    national = (
        events.loc[events["scope"] == "national"]
        .groupby("date")
        .size()
        .rename("national_news")
        .reset_index()
    )
    regional = events.loc[events["scope"] == "regional"].copy()
    regions = dictionary[["territory_id", "region"]].drop_duplicates()
    rows = []
    for _, event in regional.iterrows():
        key = str(event["region_key"]).lower().replace("ё", "е")
        hit = regions.loc[region_mask(regions["region"], key), "territory_id"]
        for territory_id in hit:
            rows.append({"territory_id": int(territory_id), "date": event["date"], "hit": 1})
    if rows:
        regional_counts = (
            pd.DataFrame(rows).groupby(["territory_id", "date"], as_index=False)["hit"].sum()
            .rename(columns={"hit": "regional_news"})
        )
    else:
        regional_counts = pd.DataFrame(columns=["territory_id", "date", "regional_news"])
    return national, regional_counts


def build_panel(cfg: dict, raw: Path, dictionary: pd.DataFrame) -> pd.DataFrame:
    consumption = pd.read_parquet(raw / "consumption.parquet")
    consumption = consumption.drop(columns=[c for c in consumption.columns if str(c).startswith("__")], errors="ignore")
    consumption = _rename_categories(consumption)
    consumption["date"] = month_start(consumption["date"])
    consumption["territory_id"] = consumption["territory_id"].astype(int)
    wide = (
        consumption.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="mean")
        .reset_index()
    )
    parts = ["food", "health", "cafe", "transport", "market"]
    for column in parts:
        if column not in wide.columns:
            wide[column] = np.nan
    part_sum = wide[parts].sum(axis=1).replace(0, np.nan)
    for column in parts:
        wide[f"share_{column}"] = wide[column] / part_sum
    wide["parts_sum"] = wide[parts].sum(axis=1)

    access = pd.read_parquet(raw / "market_access.parquet")
    access["territory_id"] = access["territory_id"].astype(int)
    wide = wide.merge(access, on="territory_id", how="left")
    wide["market_access"] = wide["market_access"].fillna(wide["market_access"].median())

    print("считаю соседей по шоссе")
    neigh = _neighbor_means(raw, wide, cfg["neighbors"]["k"], cfg["neighbors"]["max_km"])
    wide = wide.merge(neigh, on=["territory_id", "date"], how="left")
    national = wide.groupby("date")["total"].transform("mean")
    wide["neighbor_mean"] = wide["neighbor_mean"].fillna(national)
    wide["national_mean"] = national

    wide = wide.merge(dictionary[["territory_id", "name", "region"]], on="territory_id", how="left")
    wide["name"] = wide["name"].fillna("")
    wide["region"] = wide["region"].fillna("")

    wide = wide.merge(working_day_table(), on="date", how="left")
    rate = pd.read_csv(ROOT / "data_static" / "key_rate.csv")
    cpi = pd.read_csv(ROOT / "data_static" / "cpi.csv")
    rate["date"] = month_start(rate["month"])
    cpi["date"] = month_start(cpi["month"])
    wide = wide.merge(rate[["date", "rate"]], on="date", how="left")
    wide = wide.merge(cpi[["date", "cpi_yoy"]], on="date", how="left")
    wide = wide.rename(columns={"rate": "key_rate"})

    national_news, regional_news = _news_features(dictionary)
    wide = wide.merge(national_news, on="date", how="left")
    wide = wide.merge(regional_news, on=["territory_id", "date"], how="left")
    wide["national_news"] = wide["national_news"].fillna(0).astype(int)
    wide["regional_news"] = wide["regional_news"].fillna(0).astype(int)
    wide["total_obs"] = wide["total"]

    months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    ids = wide["territory_id"].unique()
    full_index = pd.MultiIndex.from_product([ids, months], names=["territory_id", "date"])
    wide = wide.set_index(["territory_id", "date"])
    wide = wide[~wide.index.duplicated(keep="first")].reindex(full_index).reset_index()
    for column in ["market_access", "name", "region"]:
        wide[column] = wide.groupby("territory_id")[column].ffill()
        wide[column] = wide.groupby("territory_id")[column].bfill()
    wide["total"] = wide.groupby("territory_id")["total_obs"].transform(lambda s: s.interpolate(limit=1))
    nat = wide.groupby("date")["total_obs"].transform("mean")
    wide["national_mean"] = nat
    wide["neighbor_mean"] = wide["neighbor_mean"].fillna(wide["national_mean"])
    for column in ["key_rate", "cpi_yoy", "n_working_days", "national_news"]:
        by_date = wide.groupby("date")[column].transform("median")
        wide[column] = wide[column].fillna(by_date)
    wide["regional_news"] = wide["regional_news"].fillna(0)
    wide["name"] = wide["name"].fillna("")
    wide["region"] = wide["region"].fillna("")

    wide = wide.sort_values(["territory_id", "date"]).reset_index(drop=True)
    processed = ROOT / cfg["data"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(processed / "panel.parquet", index=False)

    profile = {
        "n_rows": int(len(wide)),
        "n_territories": int(wide["territory_id"].nunique()),
        "date_min": str(wide["date"].min().date()),
        "date_max": str(wide["date"].max().date()),
        "n_named": int((wide.drop_duplicates("territory_id")["name"] != "").sum()),
        "total_mean": float(wide["total"].mean()),
        "total_median": float(wide["total"].median()),
        "share_means": {c: float(wide[f"share_{c}"].mean()) for c in parts},
        "region_examples": [
            str(v) for v in wide.drop_duplicates("territory_id")["region"].dropna().unique()[:8]
        ],
    }
    (processed / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"панель: {profile['n_territories']} МО, {profile['date_min']} — {profile['date_max']}, "
        f"названия есть у {profile['n_named']}"
    )
    return wide
