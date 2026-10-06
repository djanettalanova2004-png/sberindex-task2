"""Скачивание архива СберИндекса и справочника муниципалитетов."""

from __future__ import annotations

import shutil
import sqlite3
import ssl
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEVEN = Path(r"C:\Program Files\7-Zip\7z.exe")


def _ssl_context() -> ssl.SSLContext:
    return ssl._create_unverified_context()


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"уже скачан: {dest.name}")
        return dest
    print(f"скачиваю {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180, context=_ssl_context()) as response:
        dest.write_bytes(response.read())
    print(f"сохранено {dest} ({dest.stat().st_size} байт)")
    return dest


def extract_zip(zip_path: Path, dest: Path) -> None:
    marker = dest / "consumption.parquet"
    if marker.exists():
        print("архив расходов уже распакован")
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(dest))
    # файлы лежат во вложенной папке
    nested = list(dest.rglob("consumption.parquet"))
    if not nested:
        raise FileNotFoundError("в архиве нет consumption.parquet")
    folder = nested[0].parent
    if folder != dest:
        for item in folder.iterdir():
            target = dest / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))


def extract_rar(rar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*")) and any(p.is_file() for p in dest.rglob("*")):
        print("справочник уже распакован")
        return
    seven = SEVEN if SEVEN.exists() else shutil.which("7z")
    if not seven:
        raise RuntimeError("Не найден 7-Zip, он нужен чтобы распаковать справочник .rar")
    subprocess.run([str(seven), "x", "-y", f"-o{dest}", str(rar_path)], check=True)


def _read_gpkg(path: Path) -> list[pd.DataFrame]:
    frames = []
    con = sqlite3.connect(path)
    try:
        tables = pd.read_sql(
            "SELECT table_name FROM gpkg_contents", con
        )["table_name"].tolist()
    except Exception:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
    for name in tables:
        if name.startswith("gpkg_") or name.startswith("rtree") or name.startswith("sqlite"):
            continue
        try:
            info = pd.read_sql(f'PRAGMA table_info("{name}")', con)
        except Exception:
            continue
        keep = [
            column
            for column in info["name"].tolist()
            if str(column).lower() not in {"geom", "geometry", "shape", "wkb_geometry"}
        ]
        if not keep:
            continue
        col_sql = ", ".join(f'"{column}"' for column in keep)
        try:
            frame = pd.read_sql(f'SELECT {col_sql} FROM "{name}"', con)
        except Exception:
            continue
        frame["_source_table"] = name
        frames.append(frame)
    con.close()
    return frames


def _read_any(path: Path) -> list[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        frame = pd.read_csv(path, sep=None, engine="python")
        return [frame]
    if suffix in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        return [book.parse(sheet) for sheet in book.sheet_names]
    if suffix == ".parquet":
        return [pd.read_parquet(path)]
    if suffix == ".gpkg":
        return _read_gpkg(path)
    return []


def _pick_dictionary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    candidates = []
    for frame in frames:
        cols = {c.lower(): c for c in frame.columns}
        if "territory_id" not in cols:
            continue
        renamed = frame.rename(columns={v: k for k, v in cols.items()})
        candidates.append(renamed)
    if not candidates:
        raise RuntimeError("в справочнике нет колонки territory_id")
    # берём таблицу, где больше всего разных территорий и есть название
    def score(frame: pd.DataFrame) -> tuple:
        name_cols = [c for c in frame.columns if "name" in c or "назв" in c]
        region_cols = [c for c in frame.columns if "region" in c or "subject" in c or "регион" in c]
        return (int(frame["territory_id"].nunique()), len(name_cols), len(region_cols), len(frame))

    best = max(candidates, key=score)
    return best


def _column(frame: pd.DataFrame, needles: list[str]) -> str | None:
    for col in frame.columns:
        low = str(col).lower()
        if any(n in low for n in needles):
            return col
    return None


def tidy_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["territory_id"] = pd.to_numeric(frame["territory_id"], errors="coerce")
    if "municipal_district_name" in frame.columns:
        name_col = "municipal_district_name"
    else:
        name_col = _column(frame, ["name", "назв", "municip"])
    if "region_name" in frame.columns:
        region_col = "region_name"
    else:
        region_col = _column(frame, ["region_name", "регион", "subject"])
    out["name"] = frame[name_col].astype(str) if name_col else ""
    out["region"] = frame[region_col].astype(str) if region_col else ""
    year_from = _column(frame, ["year_from", "from_year"])
    year_to = _column(frame, ["year_to", "to_year"])
    if year_from and year_to:
        yf = pd.to_numeric(frame[year_from], errors="coerce")
        yt = pd.to_numeric(frame[year_to], errors="coerce")
        mask = (yf.fillna(0) <= 2024) & (yt.fillna(9999) > 2024)
        out = out.loc[mask].copy()
    out = out.dropna(subset=["territory_id"])
    out["territory_id"] = out["territory_id"].astype(int)
    out = out.drop_duplicates("territory_id")
    return out


def load_dictionary(dict_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in dict_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".parquet", ".gpkg"}:
            continue
        try:
            frames.extend(_read_any(path))
        except Exception as exc:
            print(f"не читается {path.name}: {exc}")
    if not frames:
        raise RuntimeError(f"в {dict_dir} нет таблиц справочника")
    raw = _pick_dictionary(frames)
    tidy = tidy_dictionary(raw)
    print(f"справочник: {len(tidy)} территорий, колонки исходной таблицы: {list(raw.columns)[:12]}")
    return tidy


def ensure_raw(cfg: dict) -> tuple[Path, pd.DataFrame]:
    raw = ROOT / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = download_file(cfg["data"]["zip_url"], raw / "hackathonlicence.zip")
    extract_zip(zip_path, raw)
    rar_path = download_file(cfg["data"]["dict_url"], raw / "t_dict_municipal.rar")
    dict_dir = raw / "dictionary"
    extract_rar(rar_path, dict_dir)
    dictionary = load_dictionary(dict_dir)
    dictionary.to_parquet(raw / "dictionary.parquet", index=False)
    return raw, dictionary
