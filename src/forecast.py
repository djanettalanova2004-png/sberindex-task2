"""Прогнозы: Prophet, сезонный наивный, панельный, LightGBM, Chronos и ансамбль."""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZONS = (1, 3, 6, 12)
PROPHET_BACKEND = "nls"


def add_month(value, k: int) -> pd.Timestamp:
    return pd.Timestamp(value) + pd.DateOffset(months=int(k))


def official_jobs(cfg: dict) -> list[tuple[pd.Timestamp, list[int]]]:
    found: dict[pd.Timestamp, set[int]] = {}
    for horizon, origins in cfg["origins"].items():
        for origin in origins:
            stamp = pd.Timestamp(f"{origin}-01")
            found.setdefault(stamp, set()).add(int(horizon))
    return [(origin, sorted(horizons)) for origin, horizons in sorted(found.items())]


def residual_jobs() -> list[tuple[pd.Timestamp, list[int]]]:
    origins = pd.date_range("2023-12-01", "2024-11-01", freq="MS")
    return [(origin, [1]) for origin in origins]


def _lookup_actual(panel: pd.DataFrame) -> pd.Series:
    observed = panel.dropna(subset=["total_obs"])
    return observed.set_index(["territory_id", "date"])["total_obs"]


def _attach(rows: list[dict], actual: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = list(zip(frame["territory_id"], frame["target"]))
    frame["y"] = [actual.get(key, np.nan) for key in keys]
    frame["yhat"] = np.clip(frame["yhat"].astype(float), 0, None)
    return frame


def naive_and_panel(panel: pd.DataFrame, jobs: list[tuple[pd.Timestamp, list[int]]]) -> pd.DataFrame:
    actual = _lookup_actual(panel)
    level = panel.dropna(subset=["total_obs"]).groupby("date")["total_obs"].mean()
    ids = panel["territory_id"].drop_duplicates().to_numpy()
    by_id = {
        territory_id: group.set_index("date")["total_obs"]
        for territory_id, group in panel.groupby("territory_id")
    }
    rows = []
    for origin, horizons in jobs:
        prev = add_month(origin, -12)
        growth = 1.0
        if prev in level.index and origin in level.index and pd.notna(level.loc[prev]) and float(level.loc[prev]) != 0:
            growth = float(level.loc[origin] / level.loc[prev])
        for horizon in horizons:
            target = add_month(origin, horizon)
            anchor = add_month(target, -12)
            for territory_id in ids:
                series = by_id[territory_id]
                y_anchor = series.get(anchor, np.nan)
                if pd.isna(y_anchor):
                    continue
                base = {"territory_id": int(territory_id), "origin": origin, "horizon": int(horizon), "target": target}
                rows.append({**base, "model": "seasonal_naive", "yhat": float(y_anchor)})
                rows.append({**base, "model": "panel_seasonal", "yhat": float(y_anchor) * growth})
    return _attach(rows, actual)


def prophet_backend() -> str:
    """Возвращает stan, если библиотека Prophet реально поднимает CmdStan, иначе nls."""
    try:
        from prophet import Prophet

        Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            n_changepoints=1,
            uncertainty_samples=0,
        )
        return "stan"
    except Exception as exc:
        print(f"CmdStan недоступен ({type(exc).__name__}: {exc}). Спецификация Prophet оценивается нелинейным МНК.")
        return "nls"


def prophet_note(backend: str) -> str:
    if backend == "stan":
        return (
            "Отдельная модель Prophet на каждый муниципалитет: мультипликативная годовая сезонность "
            "порядка 3 и одна точка излома. Оценка через CmdStan."
        )
    return (
        "Отдельная модель на каждый муниципалитет со спецификацией Prophet: кусочно-линейный тренд "
        "с одной точкой излома в первых 80% истории и мультипликативная годовая сезонность Фурье порядка 3. "
        "Библиотека Prophet установлена, но CmdStan на этой машине не собрался — нет компилятора mingw32-make. "
        "Параметры найдены нелинейным методом наименьших квадратов на ряде, нормированном на максимум модуля, "
        "со штрафами changepoint_prior_scale=0.05 и seasonality_prior_scale=10, как в Prophet по умолчанию."
    )


def _fourier(dates: pd.DatetimeIndex, period: float, order: int) -> np.ndarray:
    days = (pd.DatetimeIndex(dates) - pd.Timestamp("1970-01-01")).total_seconds().to_numpy() / 86400.0
    columns = []
    for harmonic in range(1, order + 1):
        angle = 2 * np.pi * harmonic * days / period
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns) if columns else np.zeros((len(dates), 0))


def _piecewise_trend(t: np.ndarray, changepoints: np.ndarray, k: float, m: float, delta: np.ndarray) -> np.ndarray:
    if len(changepoints) == 0:
        return k * t + m
    above = (t[:, None] >= changepoints[None, :]).astype(float)
    gamma = -changepoints * delta
    return (k + above @ delta) * t + (m + above @ gamma)


def _prophet_nls(hist: pd.Series, future: list[pd.Timestamp], fourier_order: int, n_changepoints: int) -> dict[pd.Timestamp, float]:
    from scipy.optimize import least_squares

    dates = pd.DatetimeIndex(hist.index)
    y = hist.to_numpy(dtype=float)
    y_scale = float(np.max(np.abs(y))) or 1.0
    scaled = y / y_scale
    start = dates.min()
    t_scale = max(int((dates.max() - start).days), 1)
    t = np.array([(stamp - start).days / t_scale for stamp in dates], dtype=float)
    hist_size = int(np.floor(len(dates) * 0.8))
    n_cp = int(n_changepoints)
    if hist_size < 2:
        n_cp = 0
    else:
        n_cp = min(n_cp, hist_size - 1)
    if n_cp > 0:
        indexes = np.linspace(0, hist_size - 1, n_cp + 1).round().astype(int)[1:]
        changepoints = t[np.clip(indexes, 0, len(t) - 1)]
    else:
        changepoints = np.array([], dtype=float)
    fourier = _fourier(dates, 365.25, fourier_order)
    slope, intercept = np.polyfit(t, scaled, 1)
    theta0 = np.concatenate([[slope, intercept], np.zeros(n_cp + fourier.shape[1])])
    sigma = max(float(np.std(scaled - (slope * t + intercept))), 0.02)

    def residual(theta: np.ndarray) -> np.ndarray:
        k, m = float(theta[0]), float(theta[1])
        delta = theta[2 : 2 + n_cp]
        beta = theta[2 + n_cp :]
        trend = _piecewise_trend(t, changepoints, k, m, delta)
        season = fourier @ beta if fourier.size else 0.0
        yhat = trend * (1.0 + season)
        data = (scaled - yhat) / sigma
        penalty = np.concatenate([delta / 0.05, beta / 10.0]) if len(theta) > 2 else np.array([])
        return np.concatenate([data, penalty])

    fitted = least_squares(residual, theta0, method="trf", max_nfev=80)
    theta = fitted.x
    future_index = pd.DatetimeIndex(future)
    t_future = np.array([(stamp - start).days / t_scale for stamp in future_index], dtype=float)
    fourier_future = _fourier(future_index, 365.25, fourier_order)
    k, m = float(theta[0]), float(theta[1])
    delta = theta[2 : 2 + n_cp]
    beta = theta[2 + n_cp :]
    trend = _piecewise_trend(t_future, changepoints, k, m, delta)
    season = fourier_future @ beta if fourier_future.size else 0.0
    yhat = np.clip(trend * (1.0 + season) * y_scale, 0, None)
    return {pd.Timestamp(stamp): float(value) for stamp, value in zip(future_index, yhat)}


def _prophet_task(task: tuple) -> list[tuple]:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    territory_id, months, values, jobs, fourier_order, n_changepoints, uncertainty_samples, backend = task
    dates = pd.to_datetime(months)
    series = pd.Series(values, index=dates, dtype=float)
    out = []
    stan_model = None
    if backend == "stan":
        import logging

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        from prophet import Prophet

        stan_model = Prophet
    for origin_str, horizons in jobs:
        origin = pd.Timestamp(origin_str)
        hist = series.loc[series.index <= origin].dropna()
        future = [add_month(origin, h) for h in horizons]
        if len(hist) < 10 or hist.nunique() < 2:
            for horizon, target in zip(horizons, future):
                anchor = add_month(target, -12)
                yhat = hist.get(anchor, np.nan)
                out.append((territory_id, origin, int(horizon), target, None if pd.isna(yhat) else float(yhat)))
            continue
        try:
            if backend == "stan":
                model = stan_model(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    n_changepoints=n_changepoints,
                    uncertainty_samples=uncertainty_samples,
                    seasonality_mode="multiplicative",
                )
                model.add_seasonality(name="yearly", period=365.25, fourier_order=fourier_order)
                model.fit(pd.DataFrame({"ds": hist.index, "y": hist.to_numpy()}))
                forecast = model.predict(pd.DataFrame({"ds": future}))
                predicted = dict(zip(pd.to_datetime(forecast["ds"]), forecast["yhat"].astype(float)))
            else:
                predicted = _prophet_nls(hist, future, int(fourier_order), int(n_changepoints))
            for horizon, target in zip(horizons, future):
                yhat = predicted.get(pd.Timestamp(target), np.nan)
                out.append((territory_id, origin, int(horizon), target, None if pd.isna(yhat) else float(yhat)))
        except Exception as exc:
            print(f"Prophet fallback {territory_id} {origin.date()}: {type(exc).__name__}: {exc}")
            for horizon, target in zip(horizons, future):
                anchor = add_month(target, -12)
                yhat = hist.get(anchor, np.nan)
                out.append((territory_id, origin, int(horizon), target, None if pd.isna(yhat) else float(yhat)))
    return out


def prophet_forecast(panel: pd.DataFrame, jobs: list[tuple[pd.Timestamp, list[int]]], cfg: dict) -> pd.DataFrame:
    global PROPHET_BACKEND
    backend = prophet_backend()
    PROPHET_BACKEND = backend
    observed = panel.dropna(subset=["total_obs"])
    packed = []
    job_payload = [(str(origin.date()), horizons) for origin, horizons in jobs]
    for territory_id, group in observed.groupby("territory_id"):
        group = group.sort_values("date")
        packed.append(
            (
                int(territory_id),
                [d.strftime("%Y-%m-%d") for d in group["date"]],
                group["total_obs"].astype(float).tolist(),
                job_payload,
                cfg["prophet"]["fourier_order"],
                cfg["prophet"]["n_changepoints"],
                cfg["prophet"]["uncertainty_samples"],
                backend,
            )
        )
    workers = min(int(cfg["prophet"]["max_workers"]), max(1, os.cpu_count() or 1))
    print(f"Prophet ({backend}): {len(packed)} рядов, {workers} процессов")
    rows_raw = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(_prophet_task, packed, chunksize=20):
            rows_raw.extend(batch)
            done += 1
            if done % 200 == 0 or done == len(packed):
                print(f"  Prophet {done}/{len(packed)}")
    actual = _lookup_actual(panel)
    rows = [
        {
            "territory_id": territory_id,
            "origin": pd.Timestamp(origin),
            "horizon": horizon,
            "target": pd.Timestamp(target),
            "model": "prophet",
            "yhat": np.nan if yhat is None else yhat,
        }
        for territory_id, origin, horizon, target, yhat in rows_raw
    ]
    return _attach(rows, actual)


def _with_lags(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["territory_id", "date"]).copy()
    grouped = frame.groupby("territory_id", sort=False)["total_obs"]
    frame["lag1"] = frame["total_obs"]
    frame["lag2"] = grouped.shift(1)
    frame["lag3"] = grouped.shift(2)
    frame["lag6"] = grouped.shift(5)
    frame["roll3"] = grouped.transform(lambda s: s.rolling(3, min_periods=3).mean())
    frame["roll6"] = grouped.transform(lambda s: s.rolling(6, min_periods=6).mean())
    return frame


def _lgbm_matrix(frame: pd.DataFrame, horizon: int, work_days: dict) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("territory_id", sort=False)["total_obs"]
    out["seasonal_anchor"] = grouped.shift(12 - horizon)
    out["y"] = grouped.shift(-horizon)
    out["target_date"] = out["date"] + pd.DateOffset(months=horizon)
    month = out["target_date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["n_working_days_target"] = out["target_date"].map(work_days)
    return out


LGBM_FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "lag6",
    "roll3",
    "roll6",
    "seasonal_anchor",
    "share_food",
    "share_health",
    "share_cafe",
    "share_transport",
    "share_market",
    "market_access",
    "neighbor_mean",
    "national_mean",
    "key_rate",
    "cpi_yoy",
    "national_news",
    "regional_news",
    "month_sin",
    "month_cos",
    "n_working_days_target",
]


def lgbm_forecast(panel: pd.DataFrame, jobs: list[tuple[pd.Timestamp, list[int]]], cfg: dict) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    lagged = _with_lags(panel)
    work_days = panel.groupby("date")["n_working_days"].median().to_dict()
    actual = _lookup_actual(panel)
    rows = []
    params = cfg["lgbm"]
    for origin, horizons in jobs:
        for horizon in horizons:
            built = _lgbm_matrix(lagged, horizon, work_days)
            train = built.loc[built["target_date"] <= origin].dropna(subset=["y", "lag1", "lag2", "lag3", "lag6"])
            test = built.loc[built["date"] == origin].copy()
            if train.empty or test.empty:
                print(f"  LightGBM {origin.date()} h={horizon}: нет обучающих пар, пропуск")
                continue
            model = LGBMRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                random_state=42,
                verbose=-1,
                n_jobs=4,
            )
            model.fit(train[LGBM_FEATURES], train["y"])
            test["yhat"] = model.predict(test[LGBM_FEATURES])
            for record in test[["territory_id", "yhat", "target_date"]].itertuples(index=False):
                rows.append(
                    {
                        "territory_id": int(record.territory_id),
                        "origin": origin,
                        "horizon": int(horizon),
                        "target": pd.Timestamp(record.target_date),
                        "model": "lightgbm",
                        "yhat": float(record.yhat),
                    }
                )
            print(f"  LightGBM {origin.date()} h={horizon}: train {len(train)}")
    return _attach(rows, actual)


def _silence_ssl() -> None:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import urllib3

        urllib3.disable_warnings()
    except Exception:
        pass
    try:
        import requests

        original = requests.Session.request

        def wrapped(self, method, url, **kwargs):
            kwargs["verify"] = False
            return original(self, method, url, **kwargs)

        requests.Session.request = wrapped
    except Exception:
        pass
    try:
        import httpx

        original_init = httpx.Client.__init__

        def client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            return original_init(self, *args, **kwargs)

        httpx.Client.__init__ = client_init
    except Exception:
        pass


def _load_chronos(model_id: str):
    _silence_ssl()
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipeline


def _median_forecast(pipeline, contexts: list, prediction_length: int) -> np.ndarray:
    import torch

    tensors = [torch.tensor(series, dtype=torch.float32) for series in contexts]
    if hasattr(pipeline, "predict_quantiles"):
        quantiles, _mean = pipeline.predict_quantiles(
            tensors, prediction_length=prediction_length, quantile_levels=[0.5]
        )
        array = quantiles.detach().cpu().numpy()
        return array[:, :, 0] if array.ndim == 3 else array
    forecast = pipeline.predict(tensors, prediction_length=prediction_length)
    array = forecast.detach().cpu().numpy()
    if array.ndim == 3:
        mid = array.shape[1] // 2
        return array[:, mid, :]
    return array


def chronos_forecast(panel: pd.DataFrame, jobs: list[tuple[pd.Timestamp, list[int]]], cfg: dict) -> pd.DataFrame:
    os.environ.setdefault("HF_HOME", str((__import__("pathlib").Path(__file__).resolve().parents[1] / ".hf_cache")))
    pipeline = _load_chronos(cfg["chronos"]["model_id"])
    observed = panel.sort_values(["territory_id", "date"])
    series_map = {
        int(territory_id): group.dropna(subset=["total_obs"]).set_index("date")["total_obs"].astype(float)
        for territory_id, group in observed.groupby("territory_id")
    }
    ids = np.array(sorted(series_map))
    actual = _lookup_actual(panel)
    batch_size = int(cfg["chronos"]["batch_size"])
    rows = []
    for origin, horizons in jobs:
        pred_len = max(horizons)
        contexts = []
        kept = []
        for territory_id in ids:
            history = series_map[territory_id]
            history = history.loc[history.index <= origin].dropna()
            if len(history) < 6:
                continue
            contexts.append(history.to_numpy())
            kept.append(int(territory_id))
        print(f"  Chronos {origin.date()} h={horizons}: {len(kept)} рядов")
        pieces = []
        for start in range(0, len(contexts), batch_size):
            pieces.append(_median_forecast(pipeline, contexts[start : start + batch_size], pred_len))
        if not pieces:
            continue
        predicted = np.vstack(pieces)
        for index, territory_id in enumerate(kept):
            for horizon in horizons:
                rows.append(
                    {
                        "territory_id": territory_id,
                        "origin": origin,
                        "horizon": int(horizon),
                        "target": add_month(origin, horizon),
                        "model": "chronos",
                        "yhat": float(predicted[index, horizon - 1]),
                    }
                )
    return _attach(rows, actual)


def ensemble(forecasts: pd.DataFrame) -> pd.DataFrame:
    parts = forecasts.loc[forecasts["model"].isin(["lightgbm", "chronos"])]
    if parts.empty:
        return parts
    pivot = parts.pivot_table(index=["territory_id", "origin", "horizon", "target", "y"], columns="model", values="yhat")
    pivot["yhat"] = pivot.median(axis=1)
    out = pivot.reset_index()[["territory_id", "origin", "horizon", "target", "y", "yhat"]]
    out["model"] = "ensemble"
    out["yhat"] = np.clip(out["yhat"].astype(float), 0, None)
    return out


def metrics_table(forecasts: pd.DataFrame, jobs: list[tuple[pd.Timestamp, list[int]]]) -> pd.DataFrame:
    allowed = {(origin, horizon) for origin, horizons in jobs for horizon in horizons}
    frame = forecasts.copy()
    frame = frame.loc[[(o, h) in allowed for o, h in zip(frame["origin"], frame["horizon"])]]
    frame = frame.dropna(subset=["y", "yhat"])
    rows = []
    for (model, horizon), group in frame.groupby(["model", "horizon"]):
        rows.append({"model": model, "horizon": int(horizon), **_scores(group["y"], group["yhat"]), "n": int(len(group))})
    by_h = pd.DataFrame(rows)
    overall = []
    for model, group in by_h.groupby("model"):
        overall.append(
            {
                "model": model,
                "horizon": 0,
                "mae": float(group["mae"].mean()),
                "r2": float(group["r2"].mean()),
                "smape": float(group["smape"].mean()),
                "n": int(group["n"].sum()),
                "n_horizons": int(group["horizon"].nunique()),
            }
        )
    return pd.concat([by_h, pd.DataFrame(overall)], ignore_index=True)


def _scores(y: pd.Series, yhat: pd.Series) -> dict:
    y = y.to_numpy(dtype=float)
    yhat = yhat.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y - yhat)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    smape = float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8)))
    return {"mae": mae, "r2": r2, "smape": smape}


def choose_best(metrics: pd.DataFrame) -> str:
    overall = metrics.loc[metrics["horizon"] == 0].copy()
    full = overall.loc[overall["n_horizons"] >= 4]
    pool = full if not full.empty else overall
    prophet = pool.loc[pool["model"] == "prophet", "mae"]
    prophet_mae = float(prophet.iloc[0]) if len(prophet) else np.inf
    contenders = pool.loc[pool["model"] != "prophet"].sort_values("mae")
    better = contenders.loc[contenders["mae"] < prophet_mae]
    if not better.empty:
        return str(better.iloc[0]["model"])
    return str(pool.sort_values("mae").iloc[0]["model"])
