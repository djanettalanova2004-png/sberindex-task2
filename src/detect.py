"""Сравнение детекторов структурных сдвигов."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.prepare import region_mask


def onestep(forecasts: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = forecasts.loc[(forecasts["model"] == model) & (forecasts["horizon"] == 1)].copy()
    frame = frame.dropna(subset=["y", "yhat"])
    frame["date"] = pd.to_datetime(frame["target"])
    frame["resid"] = frame["y"] - frame["yhat"]
    frame = frame.loc[(frame["date"] >= "2024-01-01") & (frame["date"] <= "2024-12-01")]
    return frame[["territory_id", "date", "y", "yhat", "resid"]]


def _month_index(start: str = "2023-01-01") -> dict[pd.Timestamp, int]:
    months = pd.date_range(start, "2024-12-01", freq="MS")
    return {pd.Timestamp(month): i for i, month in enumerate(months)}


def _zscore(frame: pd.DataFrame, value: str = "resid") -> pd.Series:
    def apply(series: pd.Series) -> pd.Series:
        med = np.median(series)
        mad = np.median(np.abs(series - med)) + 1e-6
        return (series - med) / (1.4826 * mad)

    return frame.groupby("date")[value].transform(apply)


def cusum_path(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    baseline = values[:2]
    center = float(np.mean(baseline))
    scale = float(np.std(baseline)) if len(baseline) > 1 else 1.0
    drift = 0.5 * max(scale, 1e-6)
    pos = 0.0
    neg = 0.0
    stats = []
    for value in values - center:
        pos = max(0.0, pos + value - drift)
        neg = min(0.0, neg + value + drift)
        stats.append(max(pos, -neg))
    return np.asarray(stats)


def bocpd_path(values: np.ndarray, variance: float, hazard: float = 0.08) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    length = len(values)
    run = np.zeros((length + 1, length + 1))
    run[0, 0] = 1.0
    scores = np.zeros(length)
    mu0 = float(np.mean(values[:2]))
    for t in range(length):
        pred = np.empty(t + 1)
        for rlen in range(t + 1):
            mean = mu0 if rlen == 0 or t == 0 else float(np.mean(values[t - rlen : t]))
            pred[rlen] = np.exp(-0.5 * (values[t] - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)
        growth = run[t, : t + 1] * pred * (1 - hazard)
        change = float(np.sum(run[t, : t + 1] * pred * hazard))
        run[t + 1, 0] = change
        run[t + 1, 1 : t + 2] = growth
        total = run[t + 1].sum()
        if total > 0:
            run[t + 1] /= total
        scores[t] = run[t + 1, 0]
    return scores


def _hit_and_delay(alarms: list[pd.Timestamp], shock: pd.Timestamp, window: int) -> tuple[bool, float | None]:
    start = shock - pd.DateOffset(months=1)
    end = shock + pd.DateOffset(months=window)
    inside = [alarm for alarm in alarms if start <= alarm <= end and alarm >= shock - pd.DateOffset(months=1)]
    # попадание: тревога в месяце шока, за месяц до него или в пределах окна после
    useful = [alarm for alarm in alarms if shock - pd.DateOffset(months=1) <= alarm <= shock + pd.DateOffset(months=window)]
    if not useful:
        return False, None
    delay = min((alarm.year - shock.year) * 12 + (alarm.month - shock.month) for alarm in useful)
    return True, float(delay)


def _fpr_tpr(alarms_by_id, shocked, controls, shock_of, window, eval_start, eval_end):
    fp = 0
    for territory_id in controls:
        alarms = [a for a in alarms_by_id.get(territory_id, []) if eval_start <= a <= eval_end]
        fp += int(len(alarms) > 0)
    hits = []
    delays = []
    for territory_id in shocked:
        ok, delay = _hit_and_delay(alarms_by_id.get(territory_id, []), shock_of[territory_id], window)
        hits.append(ok)
        if delay is not None:
            delays.append(delay)
    fpr = fp / max(len(controls), 1)
    tpr = float(np.mean(hits)) if hits else 0.0
    precision = (sum(hits) / (sum(hits) + fp)) if (sum(hits) + fp) else 0.0
    med_delay = float(np.median(delays)) if delays else None
    return fpr, tpr, precision, med_delay


def _alarms_from_scores(scores: dict[int, np.ndarray], months: list[pd.Timestamp], threshold: float) -> dict[int, list]:
    alarms = {}
    for territory_id, path in scores.items():
        alarms[territory_id] = [month for month, value in zip(months, path) if value >= threshold]
    return alarms


def _calibrate(scores, shocked, controls, shock_of, months, window, eval_start, eval_end, target_fpr):
    pool = np.concatenate([scores[i] for i in list(shocked) + list(controls)])
    thresholds = np.unique(np.quantile(pool, np.linspace(0.5, 0.995, 40)))
    best = None
    for threshold in thresholds:
        alarms = _alarms_from_scores(scores, months, float(threshold))
        fpr, tpr, precision, delay = _fpr_tpr(alarms, shocked, controls, shock_of, window, eval_start, eval_end)
        within = fpr <= target_fpr + 0.02
        rank = (int(within), tpr if within else -fpr, -(delay if delay is not None else 99), -abs(fpr - target_fpr))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), fpr, tpr, precision, delay, within)
    return {
        "threshold": best[1],
        "fpr": best[2],
        "tpr": best[3],
        "precision": best[4],
        "median_delay": best[5],
        "within_fpr_budget": bool(best[6]),
    }


def _pelt_breakpoints(series: np.ndarray, penalty: float) -> list[int]:
    import ruptures as rpt

    series = np.asarray(series, dtype=float).reshape(-1, 1)
    if np.isnan(series).any():
        series = pd.Series(series.ravel()).interpolate(limit_direction="both").to_numpy().reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=2).fit(series)
    points = algo.predict(pen=penalty)
    return [int(point) for point in points[:-1]]


def _calibrate_pelt(series_map, month_labels, shocked, controls, shock_of, window, eval_start, eval_end, target_fpr):
    penalties = np.logspace(0, 4, 16)
    best = None
    for penalty in penalties:
        alarms = {}
        for territory_id, series in series_map.items():
            try:
                points = _pelt_breakpoints(series, float(penalty))
            except Exception:
                points = []
            alarms[territory_id] = [month_labels[i] for i in points if i < len(month_labels)]
        fpr, tpr, precision, delay = _fpr_tpr(alarms, shocked, controls, shock_of, window, eval_start, eval_end)
        within = fpr <= target_fpr + 0.02
        rank = (int(within), tpr if within else -fpr, -(delay if delay is not None else 99), -abs(fpr - target_fpr))
        if best is None or rank > best[0]:
            best = (rank, float(penalty), fpr, tpr, precision, delay, within)
    return {
        "threshold": best[1],
        "fpr": best[2],
        "tpr": best[3],
        "precision": best[4],
        "median_delay": best[5],
        "within_fpr_budget": bool(best[6]),
    }


def run_detection(panel: pd.DataFrame, forecasts: pd.DataFrame, model: str, cfg: dict) -> dict:
    step = onestep(forecasts, model)
    if step.empty:
        raise RuntimeError(f"нет помесячных остатков модели {model}")
    counts = step.groupby("territory_id").size()
    complete = counts[counts >= 12].index.to_numpy().copy()
    rng = np.random.default_rng(cfg["detection"]["seed"])
    rng.shuffle(complete)
    n_shocked = min(cfg["detection"]["n_shocked"], len(complete) // 3)
    n_control = min(cfg["detection"]["n_control"], len(complete) - n_shocked)
    shocked = complete[:n_shocked]
    controls = complete[n_shocked : n_shocked + n_control]
    shock_months = [pd.Timestamp(f"{m}-01") for m in cfg["detection"]["shock_months"]]
    shock_of = {int(territory_id): shock_months[i % len(shock_months)] for i, territory_id in enumerate(shocked)}
    shocked = [int(v) for v in shocked]
    controls = [int(v) for v in controls]

    base = step.copy()
    injected = base.set_index(["territory_id", "date"])["resid"].to_dict()
    level = base.set_index(["territory_id", "date"])["y"].to_dict()
    for territory_id, shock in shock_of.items():
        for date, value in list(level.items()):
            if date[0] == territory_id and date[1] >= shock:
                injected[date] = injected[date] + cfg["detection"]["shock_size"] * value

    resid_rows = [{"territory_id": key[0], "date": key[1], "resid": value} for key, value in injected.items()]
    resid_frame = pd.DataFrame(resid_rows)
    resid_frame["z"] = _zscore(resid_frame)
    months = list(pd.date_range("2024-01-01", "2024-12-01", freq="MS"))
    eval_start = pd.Timestamp(cfg["detection"]["eval_start"] + "-01")
    eval_end = pd.Timestamp(cfg["detection"]["eval_end"] + "-01")
    window = int(cfg["detection"]["detect_window"])
    target_fpr = float(cfg["detection"]["target_fpr"])

    def paths(column: str) -> dict[int, np.ndarray]:
        table = resid_frame.pivot(index="territory_id", columns="date", values=column)
        out = {}
        for territory_id in shocked + controls:
            if territory_id not in table.index:
                continue
            out[territory_id] = table.loc[territory_id, months].to_numpy(dtype=float)
        return out

    z_paths = paths("z")
    resid_paths = paths("resid")
    variance = float(np.nanvar(np.concatenate([path[:2] for path in resid_paths.values()]))) + 1e-3

    results = {}
    results["panel_z"] = _calibrate(z_paths, shocked, controls, shock_of, months, window, eval_start, eval_end, target_fpr)
    cusum_scores = {tid: cusum_path(path) for tid, path in resid_paths.items()}
    results["cusum"] = _calibrate(cusum_scores, shocked, controls, shock_of, months, window, eval_start, eval_end, target_fpr)
    bocpd_scores = {tid: bocpd_path(path, variance) for tid, path in resid_paths.items()}
    results["bocpd"] = _calibrate(bocpd_scores, shocked, controls, shock_of, months, window, eval_start, eval_end, target_fpr)

    # гибрид: |z| плюс региональная новость в этом или прошлом месяце
    news = panel[["territory_id", "date", "regional_news"]].drop_duplicates()
    news_table = news.pivot(index="territory_id", columns="date", values="regional_news").reindex(columns=months).fillna(0)
    best_hybrid = None
    for alpha in (0.0, 1.0, 2.0, 4.0):
        scores = {}
        for territory_id, path in z_paths.items():
            flags = news_table.loc[territory_id].to_numpy(dtype=float) if territory_id in news_table.index else np.zeros(len(months))
            lead = np.maximum(flags, np.r_[0, flags[:-1]])
            scores[territory_id] = np.abs(path) + alpha * lead
        tuned = _calibrate(scores, shocked, controls, shock_of, months, window, eval_start, eval_end, target_fpr)
        tuned["alpha"] = alpha
        rank = (tuned["tpr"], -(tuned["median_delay"] if tuned["median_delay"] is not None else 99))
        if best_hybrid is None or rank > best_hybrid[0]:
            best_hybrid = (rank, tuned)
    results["panel_news"] = best_hybrid[1]

    # PELT по сырому ряду и по остатку
    raw_months = list(pd.date_range("2023-01-01", "2024-12-01", freq="MS"))
    observed = panel.dropna(subset=["total_obs"]).pivot(index="territory_id", columns="date", values="total_obs")
    raw_map = {}
    resid_long = {}
    for territory_id in shocked + controls:
        if territory_id in observed.index:
            series = observed.loc[territory_id, raw_months].to_numpy(dtype=float).copy()
        else:
            series = np.full(24, np.nan)
        shock = shock_of.get(territory_id)
        if shock is not None:
            for i, month in enumerate(raw_months):
                if month >= shock and np.isfinite(series[i]):
                    series[i] *= 1 + cfg["detection"]["shock_size"]
        raw_map[territory_id] = series
        resid_long[territory_id] = resid_paths[territory_id]
    print("  PELT по сырому ряду")
    results["pelt_raw"] = _calibrate_pelt(raw_map, raw_months, shocked, controls, shock_of, window, eval_start, eval_end, target_fpr)
    print("  PELT по остаткам")
    results["pelt_resid"] = _calibrate_pelt(resid_long, months, shocked, controls, shock_of, window, eval_start, eval_end, target_fpr)

    # новости сами по себе, без подбора порога
    news_alarms = {}
    for territory_id in shocked + controls:
        if territory_id not in news_table.index:
            news_alarms[territory_id] = []
            continue
        flags = news_table.loc[territory_id]
        news_alarms[territory_id] = [month for month in months if flags.get(month, 0) > 0 or flags.get(month - pd.DateOffset(months=1), 0) > 0]
    fpr, tpr, precision, delay = _fpr_tpr(news_alarms, shocked, controls, shock_of, window, eval_start, eval_end)
    results["news_only"] = {"threshold": None, "fpr": fpr, "tpr": tpr, "precision": precision, "median_delay": delay, "alpha": None}

    cases = _cases(panel, step, cfg)
    ranked = sorted(
        ((name, item) for name, item in results.items() if name != "news_only"),
        key=lambda pair: (
            int(pair[1].get("within_fpr_budget", False)),
            pair[1]["tpr"],
            -(pair[1]["median_delay"] if pair[1]["median_delay"] is not None else 99),
        ),
        reverse=True,
    )
    best_name = ranked[0][0]
    return {
        "residual_model": model,
        "best_detector": best_name,
        "methods": results,
        "cases": cases,
        "n_shocked": len(shocked),
        "n_control": len(controls),
    }


def _cases(panel: pd.DataFrame, step: pd.DataFrame, cfg: dict) -> dict:
    real = step.copy()
    real["z"] = _zscore(real)
    names = panel[["territory_id", "name", "region"]].drop_duplicates("territory_id")
    real = real.merge(names, on="territory_id", how="left")

    def region_gap(key: str, month: str) -> dict:
        stamp = pd.Timestamp(f"{month}-01")
        part = real.loc[real["date"] == stamp]
        inside_mask = region_mask(part["region"], key)
        inside = part.loc[inside_mask, "z"].abs()
        outside = part.loc[~inside_mask, "z"].abs()
        return {
            "month": month,
            "region": key,
            "n": int(inside.shape[0]),
            "mean_abs_z": None if inside.empty else float(inside.mean()),
            "outside_mean_abs_z": None if outside.empty else float(outside.mean()),
        }

    floods = [
        region_gap("оренбург", "2024-04"),
        region_gap("оренбург", "2024-05"),
        region_gap("курган", "2024-04"),
        region_gap("тюмен", "2024-04"),
        region_gap("москва", "2024-03"),
    ]
    rate_month = real.loc[real["date"] == "2024-07-01", "z"].abs()
    quiet_month = real.loc[real["date"] == "2024-02-01", "z"].abs()
    top = (
        real.loc[real["date"] == "2024-04-01"]
        .assign(abs_z=lambda frame: frame["z"].abs())
        .sort_values("abs_z", ascending=False)
        .head(5)
    )
    examples = []
    for _, row in top.iterrows():
        history = panel.loc[panel["territory_id"] == row["territory_id"], ["date", "total_obs"]].dropna()
        pred = step.loc[step["territory_id"] == row["territory_id"], ["date", "yhat", "y"]]
        examples.append(
            {
                "territory_id": int(row["territory_id"]),
                "name": str(row["name"]),
                "region": str(row["region"]),
                "abs_z_2024_04": float(row["abs_z"]),
                "actual": {str(d.date()): float(v) for d, v in zip(history["date"], history["total_obs"])},
                "forecast_2024": {str(d.date()): float(v) for d, v in zip(pred["date"], pred["yhat"])},
            }
        )
    return {
        "floods": floods,
        "rate_hike_2024_07_mean_abs_z": None if rate_month.empty else float(rate_month.mean()),
        "quiet_2024_02_mean_abs_z": None if quiet_month.empty else float(quiet_month.mean()),
        "share_abs_z_above_2_july": None if rate_month.empty else float((rate_month > 2).mean()),
        "share_abs_z_above_2_february": None if quiet_month.empty else float((quiet_month > 2).mean()),
        "examples": examples,
    }


def save_detection(result: dict, path) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
