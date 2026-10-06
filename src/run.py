"""Точка входа: данные, прогноз, детекторы, отчёт."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.detect import run_detection, save_detection
from src.download import ensure_raw
from src import forecast as forecast_mod
from src.forecast import (
    choose_best,
    chronos_forecast,
    ensemble,
    lgbm_forecast,
    metrics_table,
    naive_and_panel,
    official_jobs,
    prophet_forecast,
    prophet_note,
    residual_jobs,
)
from src.prepare import build_panel
from src.report import write_report

ROOT = Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _limit(panel: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return panel
    counts = panel.groupby("territory_id")["total_obs"].apply(lambda s: int(s.notna().sum()))
    complete = counts[counts >= 20].index.to_numpy()
    rng = __import__("numpy").random.default_rng(42)
    take = min(limit, len(complete))
    chosen = set(rng.choice(complete, size=take, replace=False).tolist())
    print(f"ограничение: {take} муниципалитетов из {panel['territory_id'].nunique()}")
    return panel.loc[panel["territory_id"].isin(chosen)].copy()


def _merge_jobs(official, extra):
    found = {origin: set(horizons) for origin, horizons in official}
    for origin, horizons in extra:
        found.setdefault(origin, set()).update(horizons)
    return [(origin, sorted(horizons)) for origin, horizons in sorted(found.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    cfg = _load_cfg()
    out = ROOT / cfg["outputs_dir"]
    out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        profile = json.loads((ROOT / cfg["data"]["processed_dir"] / "profile.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(out / "metrics.csv")
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        detection = json.loads((out / "detection.json").read_text(encoding="utf-8"))
        write_report(
            profile,
            metrics,
            summary["best_model"],
            detection,
            summary.get("chronos_note", ""),
            summary.get("prophet_note", ""),
        )
        print("отчёт обновлён")
        return

    raw, dictionary = ensure_raw(cfg)
    panel = build_panel(cfg, raw, dictionary)
    panel = _limit(panel, args.limit)
    profile = json.loads((ROOT / cfg["data"]["processed_dir"] / "profile.json").read_text(encoding="utf-8"))
    if args.limit:
        profile["n_territories"] = int(panel["territory_id"].nunique())
        profile["limited"] = True

    official = official_jobs(cfg)
    jobs = _merge_jobs(official, residual_jobs())
    pieces = [naive_and_panel(panel, jobs)]
    print("LightGBM")
    pieces.append(lgbm_forecast(panel, jobs, cfg))
    chronos_note = "Chronos-Bolt Small посчитан на CPU в режиме zero-shot."
    try:
        print("Chronos-Bolt")
        pieces.append(chronos_forecast(panel, jobs, cfg))
    except Exception as exc:
        chronos_note = f"Chronos-Bolt не запустился ({exc}). Сравнение фундаментной модели в этот прогон не вошло."
        print(chronos_note)
    print("Prophet")
    pieces.append(prophet_forecast(panel, official, cfg))
    forecasts = pd.concat(pieces, ignore_index=True)
    forecasts = pd.concat([forecasts, ensemble(forecasts)], ignore_index=True)
    forecasts.to_parquet(out / "forecasts.parquet", index=False)

    metrics = metrics_table(forecasts, official)
    best = choose_best(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print("лучшая модель:", best)

    residual_model = best
    monthly = forecasts.loc[forecasts["horizon"] == 1].groupby("model")["origin"].nunique()
    if int(monthly.get(residual_model, 0)) < 10:
        residual_model = "ensemble" if int(monthly.get("ensemble", 0)) >= 10 else "panel_seasonal"
        print(f"для остатков взята {residual_model}: у лучшей модели нет полного 2024 года")
    print("детекторы, остаток модели", residual_model)
    detection = run_detection(panel, forecasts, residual_model, cfg)
    save_detection(detection, out / "detection.json")
    summary = {
        "best_model": best,
        "residual_model": residual_model,
        "best_detector": detection["best_detector"],
        "chronos_note": chronos_note,
        "prophet_note": prophet_note(forecast_mod.PROPHET_BACKEND),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(profile, metrics, best, detection, chronos_note, summary["prophet_note"])
    print("готово:", out)


if __name__ == "__main__":
    main()
