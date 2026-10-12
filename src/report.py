"""Методологический отчёт и одностраничный лендинг."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_LABEL = {
    "prophet": "Prophet",
    "seasonal_naive": "Сезонный наивный",
    "panel_seasonal": "Панельный сезонный",
    "lightgbm": "LightGBM",
    "chronos": "Chronos-Bolt",
    "ensemble": "Ансамбль",
}
DETECTOR_LABEL = {
    "pelt_raw": "PELT по сырому ряду",
    "pelt_resid": "PELT по остаткам",
    "bocpd": "Байесовский онлайн-детектор",
    "cusum": "CUSUM по остаткам",
    "panel_z": "Панельный z-score",
    "panel_news": "Панельный z-score и новости",
    "news_only": "Только региональные новости",
}


def _paragraphs(text: str) -> str:
    parts = [html.escape(part) for part in text.split("\n\n") if part.strip()]
    return "<p>" + "</p><p>".join(parts) + "</p>"


def _num(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ")


def _spark(actual: dict, forecast: dict) -> str:
    months = sorted(actual)
    if len(months) < 2:
        return ""
    values = [actual[m] for m in months]
    forecast_points = []
    for month in months:
        key = month[:7] + "-01"
        if key in forecast:
            forecast_points.append((month, forecast[key]))
        elif month in forecast:
            forecast_points.append((month, forecast[month]))
    hi = max(values + [v for _, v in forecast_points])
    lo = min(values + [v for _, v in forecast_points])
    span = hi - lo or 1
    width, height, pad = 680, 200, 24

    def xy(i, value):
        x = pad + i * (width - 2 * pad) / (len(months) - 1)
        y = height - pad - (value - lo) * (height - 2 * pad) / span
        return x, y

    actual_pts = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(values))
    if forecast_points:
        index = {m: i for i, m in enumerate(months)}
        pred_pts = " ".join(
            f"{xy(index[m], v)[0]:.1f},{xy(index[m], v)[1]:.1f}" for m, v in forecast_points if m in index
        )
    else:
        pred_pts = ""
    return f"""<svg viewBox="0 0 {width} {height}" role="img">
      <polyline fill="none" stroke="#1c1915" stroke-width="2" points="{actual_pts}"/>
      <polyline fill="none" stroke="#0e6b45" stroke-width="2" stroke-dasharray="5 4" points="{pred_pts}"/>
    </svg>"""


def write_report(
    profile: dict,
    metrics: pd.DataFrame,
    best: str,
    detection: dict,
    chronos_note: str,
    prophet_note: str = "",
) -> None:
    report_dir = ROOT / "report"
    presentation = ROOT / "presentation"
    report_dir.mkdir(parents=True, exist_ok=True)
    presentation.mkdir(parents=True, exist_ok=True)
    text = _markdown(profile, metrics, best, detection, chronos_note, prophet_note)
    (report_dir / "methodology.md").write_text(text, encoding="utf-8")
    page = _html(profile, metrics, best, detection, chronos_note, prophet_note)
    (presentation / "index.html").write_text(page, encoding="utf-8")
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.html").write_text(page, encoding="utf-8")


def _markdown(profile, metrics, best, detection, chronos_note, prophet_note: str = "") -> str:
    by_h = metrics.loc[metrics["horizon"] > 0].copy()
    overall = metrics.loc[metrics["horizon"] == 0].sort_values("mae")
    prophet_macro = overall.loc[overall["model"] == "prophet", "mae"]
    prophet_mae = float(prophet_macro.iloc[0]) if len(prophet_macro) else float("nan")
    best_mae = float(overall.loc[overall["model"] == best, "mae"].iloc[0])
    methods = detection["methods"]
    winner = detection["best_detector"]
    lines = [
        "# Прогноз потребительских расходов муниципалитетов и раннее обнаружение шоков",
        "",
        "Краткий методологический отчёт по задаче 2 конкурса СберИндекса.",
        "",
        "## Методология",
        "",
        "Задача состоит из двух частей: прогноз расхода муниципалитета и ранний сигнал, что расход этого муниципалитета разошёлся с привычной динамикой.",
        "",
        "1. Цель — месячный ряд «все категории» по каждому муниципалитету. Пять остальных категорий, соседи по шоссе, доступность рынков, ставка, ИПЦ, рабочие дни и новости входят как признаки, а не как отдельные цели.",
        "2. Прогноз прямой. На горизонтах 1, 3, 6 и 12 месяцев модель видит только данные до даты отсечения и сразу называет значение через h месяцев, без цепочки промежуточных прогнозов.",
        "3. Модели сравниваются по MAE. Рядом считается R². База — спецификация Prophet. Лучшей считается модель, которая посчитана на всех четырёх горизонтах и имеет меньшую среднюю MAE, чем Prophet.",
        "4. Современная фундаментная модель — Chronos-Bolt Small. Она применяется без дообучения, отдельно на каждом ряду.",
        "5. Новость стыкуется с расходом по двум ключам: месяц события и география. Решение по ставке относится ко всей стране. Региональное событие — только к муниципалитетам этого субъекта из справочника СберИндекса. Будущие даты в прогноз не попадают.",
        "6. Структурный сдвиг ищется по остатку прогноза на один месяц. Методы сравниваются на искусственном скачке уровня и на реальных эпизодах 2024 года. Лучший метод — тот, у кого выше полнота при доле ложных тревог около 10% и меньше задержка.",
        "",
        "## Архитектура",
        "",
        "1. Архив СберИндекса: расходы, расстояния, доступность рынков. Справочник территорий даёт название и регион.",
        "2. Внешние ряды: ключевая ставка, годовой ИПЦ, календарь событий, число рабочих дней.",
        "3. Панель «муниципалитет × месяц» собирается в `data/processed/panel.parquet`.",
        "4. Шесть прогнозов пишутся в `outputs/forecasts.parquet`: Prophet, сезонный наивный, панельный сезонный, LightGBM, Chronos-Bolt, медиана LightGBM и Chronos.",
        "5. MAE, R² и sMAPE по горизонтам — `outputs/metrics.csv`.",
        "6. Остаток лучшей модели на один месяц подаётся в PELT, байесовский онлайн-детектор, CUSUM, панельный z-score и новостной гибрид. Итог — `outputs/detection.json`.",
        "7. Этот отчёт и страница `presentation/index.html` читают готовые файлы и не пересчитывают модели.",
        "",
        "## Что прогнозируем",
        "",
        "Берём безналичные потребительские расходы СберИндекса на уровне муниципалитетов. "
        f"В выборке {profile['n_territories']} территорий, месяцы {profile['date_min']} — {profile['date_max']}. "
        "Целевой ряд — категория «все категории». Она не равна сумме пяти остальных категорий "
        "(продовольствие, здоровье, общественное питание, транспорт, маркетплейсы), поэтому эти пять рядов используются как признаки, а не как части цели. "
        f"Среднее значение цели — {_num(profile['total_mean'], 0)}, медиана — {_num(profile['total_median'], 0)} "
        "(единицы показателя `value` в датасете СберИндекса).",
        "",
        "Дополнительно подключены расстояние по автомобильным дорогам и доступность рынков из того же архива, "
        "справочник территорий СберИндекса, ключевая ставка Банка России на конец месяца, "
        "годовой ИПЦ Росстата и календарь документированных событий.",
        "",
        "## Как проверяется прогноз",
        "",
        "Ряд длиной 24 месяца. Прогноз прямой: для горизонта 1, 3, 6 и 12 месяцев своя оценка, без рекурсии. "
        "На дате отсечения модель видит только прошлое.",
        "",
        "- 1 и 3 месяца: отсечения декабрь 2023, март, июнь и сентябрь 2024",
        "- 6 месяцев: декабрь 2023, март и июнь 2024",
        "- 12 месяцев: только декабрь 2023, цель — декабрь 2024",
        "",
        "Горизонт в 12 месяцев поэтому проверяется на одном месяце. Это ограничение данных, а не модели.",
        "",
        "Главная метрика — MAE, рядом R². Обе считаются по всем муниципалитетам внутри горизонта. "
        "Итоговая MAE модели — среднее четырёх горизонтов, чтобы короткий горизонт не перевешивал длинный. "
        "Дополнительно считается sMAPE.",
        "",
        "## Модели",
        "",
        "1. "
        + (
            prophet_note
            or "Prophet — отдельная модель на каждый муниципалитет, мультипликативная годовая сезонность порядка 3 и одна точка излома. Это обязательная база."
        ),
        "2. Сезонный наивный прогноз — то же значение, что год назад.",
        "3. Панельный сезонный прогноз — наивный прогноз, умноженный на общий рост расходов всех муниципалитетов за последний год.",
        "4. LightGBM — одна модель на все территории и на каждый горизонт. Признаки: лаги, скользящие средние, значение год назад, доли категорий, доступность рынков, средний расход восьми ближайших муниципалитетов по шоссе (не дальше 150 км), ставка, ИПЦ, число рабочих дней в целевом месяце, национальные и региональные новости. Для горизонта 12 месяцев обучающих пар нет: чтобы предсказать декабрь 2024 из декабря 2023, нужна пара «декабрь 2022 → декабрь 2023», а 2022 года в данных нет. LightGBM на этом горизонте честно не оценивается.",
        "5. Chronos-Bolt Small — предобученная фундаментная модель Amazon, zero-shot, без дообучения на этих рядах. "
        + (chronos_note or "Модель посчитана на CPU."),
        "6. Ансамбль — медиана LightGBM и Chronos там, где есть оба прогноза. На горизонте 12 месяцев остаётся Chronos.",
        "",
        "## Качество прогноза",
        "",
        _metric_markdown(by_h, overall),
        "",
        f"Лучшая модель по средней MAE горизонтов: **{MODEL_LABEL.get(best, best)}** "
        f"(MAE {_num(best_mae, 1)} против {_num(prophet_mae, 1)} у Prophet).",
        "",
        _forecast_reading(by_h, overall),
        "",
        _skill_markdown(by_h, overall),
        "",
        "## Как ищем структурный сдвиг",
        "",
        "Шок — это расход, который разошёлся с уже привычной динамикой, а не обычный декабрь. "
        f"Поэтому онлайн-детекторы смотрят на остаток прогноза модели «{MODEL_LABEL.get(detection['residual_model'], detection['residual_model'])}» "
        "на один месяц вперёд в течение 2024 года. PELT дополнительно запускается по сырому ряду: это офлайн-база, и сезонность её путает.",
        "",
        f"Синтетическая проверка: в {detection['n_shocked']} территориях с полного ряда расход с случайного месяца весны–осени 2024 умножен на 1,3. "
        f"Ещё {detection['n_control']} территорий оставлены как есть. Порог каждого метода подобран так, чтобы доля ложных тревог на контроле была около 10%. "
        "Попадание — тревога за месяц до шока или в течение трёх месяцев после. Задержка измеряется в месяцах, отрицательная задержка значит, что тревога прозвучала раньше самого скачка.",
        "",
        _detector_markdown(methods, winner),
        "",
        _detector_reading(methods, winner),
        "",
        "## Как новости стыкуются с расходами",
        "",
        "Событие привязывается к месяцу и к географии. Решение по ключевой ставке относится ко всем муниципалитетам. "
        "Региональное событие (паводок, теракт в «Крокусе») относится только к муниципалитетам субъекта, которого касается событие: "
        "название региона берётся из справочника СберИндекса и сопоставляется по ключу. "
        "В прогноз на дату отсечения попадают только события этого месяца и более ранние. Будущие новости не используются. "
        "Для раннего обнаружения региональная новость в месяце T может поднять тревогу в T и в T+1, ещё до того как остаток расхода станет большим.",
        "",
        "Отдельный детектор «только новости» не подгоняется под долю ложных тревог: у него столько тревог, сколько реальных событий в календаре. "
        "На случайных синтетических скачках он почти не попадает в цель, и это ожидаемо: синтетический скачок не написан в новостях. "
        "Польза новостей проверяется на реальных событиях ниже и внутри гибрида с панельным z-score.",
        "",
        "## Что видно на реальных событиях",
        "",
        _cases_markdown(detection["cases"]),
        "",
        "## Ограничения",
        "",
        "- Два года месячных данных. Горизонт 12 месяцев проверяется один раз.",
        "- Prophet на таком коротком ряду слабо отличает тренд от сезонности. Это свойство базы, его не нужно «улучшать» подгонкой.",
        "- ИПЦ округлён до 0,1 п.п. по публикациям Росстата, ставка — значение на конец месяца по решениям Банка России.",
        "- Календарь событий ручной и короткий. Это официальные эпизоды, а не полный поток новостей. Если бы открылся большой архив вроде GDELT, правило стыковки осталось бы тем же: регион и месяц, без будущих дат.",
        "- После декабря 2024 данных нет, поэтому «будущий» шок за границей выборки увидеть нельзя. Раннее обнаружение проверено внутри 2024 года и на синтетических скачках.",
        "",
        "## Как воспроизвести",
        "",
        "```",
        "py -3.11 -m venv .venv",
        ".\\.venv\\Scripts\\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        ".\\.venv\\Scripts\\python -m pip install -r requirements.txt",
        ".\\.venv\\Scripts\\python -m src.run",
        "```",
        "",
        "Нужен Python 3.11. Сырые parquet и справочник скрипт скачивает сам в `data/raw` и в git не кладёт. "
        "Гиперпараметры лежат в `configs/config.yaml`. "
        "Подробности про LightGBM и сборку Prophet — в `README.md`.",
        "",
    ]
    return "\n".join(lines)


def _metric_markdown(by_h: pd.DataFrame, overall: pd.DataFrame) -> str:
    horizons = [1, 3, 6, 12]
    header = "| Модель | " + " | ".join(f"{h} мес." for h in horizons) + " | средняя MAE | средний R² |"
    sep = "| --- | " + " | ".join("---" for _ in horizons) + " | --- | --- |"
    rows = [header, sep]
    prophet = by_h.loc[by_h["model"] == "prophet"].set_index("horizon")["mae"]
    for _, item in overall.sort_values("mae").iterrows():
        model = item["model"]
        cells = []
        for horizon in horizons:
            part = by_h.loc[(by_h["model"] == model) & (by_h["horizon"] == horizon)]
            if part.empty:
                cells.append("—")
            else:
                mae = float(part["mae"].iloc[0])
                r2 = float(part["r2"].iloc[0])
                cells.append(f"{_num(mae, 0)} (R² {_num(r2, 2)})")
        rows.append(
            f"| {MODEL_LABEL.get(model, model)} | "
            + " | ".join(cells)
            + f" | {_num(item['mae'], 1)} | {_num(item['r2'], 2)} |"
        )
    rows.append("")
    rows.append("В ячейке горизонта: MAE и R². Средняя MAE — среднее по тем горизонтам, где модель посчитана.")
    if len(prophet):
        rows.append("Число выше, чем у Prophet на том же горизонте, означает более слабый прогноз.")
    return "\n".join(rows)


def _detector_markdown(methods: dict, winner: str) -> str:
    lines = [
        "| Метод | Доля ложных тревог | Полнота | Точность | Медианная задержка, мес. |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = ["pelt_raw", "pelt_resid", "bocpd", "cusum", "panel_z", "panel_news", "news_only"]
    for name in order:
        item = methods.get(name)
        if not item:
            continue
        mark = " ← лучший" if name == winner else ""
        if item.get("within_fpr_budget") is False:
            mark += ", лимит ложных тревог не взят"
        delay = "—" if item["median_delay"] is None else _num(item["median_delay"], 1)
        lines.append(
            f"| {DETECTOR_LABEL.get(name, name)}{mark} | {_num(100 * item['fpr'], 1)}% | "
            f"{_num(100 * item['tpr'], 1)}% | {_num(100 * item['precision'], 1)}% | {delay} |"
        )
    return "\n".join(lines)


def _case_notes(cases: dict) -> list[str]:
    notes = [
        "Если средний |z| региона в месяц паводка близок к остальной стране, месячный расход этот шок почти не показывает. "
        "Так и нужно писать: отсутствие следа — тоже результат. "
        "Общее повышение ставки не должно раздувать панельный z-score, потому что из остатка каждого муниципалитета вычитается медиана страны."
    ]
    orenburg = next((item for item in cases["floods"] if item["region"] == "оренбург" and item["month"] == "2024-04"), None)
    moscow = next((item for item in cases["floods"] if item["region"] == "москва"), None)
    if orenburg and orenburg["n"] and orenburg["mean_abs_z"] is not None and orenburg["outside_mean_abs_z"] is not None:
        if orenburg["mean_abs_z"] <= orenburg["outside_mean_abs_z"] + 0.05:
            notes.append(
                "В апреле 2024 средний |z| Оренбургской области не выше, чем у остальной страны. "
                "Паводок в месячном ряде расходов не выделяется. Это отрицательный результат, его не нужно подтягивать."
            )
    if moscow and moscow["n"] and moscow["mean_abs_z"] is not None and moscow["outside_mean_abs_z"] is not None:
        if moscow["mean_abs_z"] > moscow["outside_mean_abs_z"] + 0.3:
            notes.append(
                "В марте 2024 Москва (без Московской области) выделяется на фоне страны. "
                "Это локальный сдвиг месяца, в котором был теракт в «Крокусе», а не общий календарный пик."
            )
    return notes


def _cases_markdown(cases: dict) -> str:
    lines = []
    for item in cases["floods"]:
        if not item["n"]:
            lines.append(
                f"- {item['region'].capitalize()}, {item['month']}: в справочнике не нашлось муниципалитетов с таким регионом."
            )
            continue
        lines.append(
            f"- {item['region'].capitalize()}, {item['month']}: средний |z| внутри региона {_num(item['mean_abs_z'], 2)}, "
            f"в остальных муниципалитетах {_num(item['outside_mean_abs_z'], 2)} "
            f"({item['n']} территорий)."
        )
    lines.append(
        f"- Июль 2024, повышение ставки до 18%: средний |z| {_num(cases['rate_hike_2024_07_mean_abs_z'], 2)}, "
        f"доля территорий с |z| > 2 равна {_num(100 * (cases['share_abs_z_above_2_july'] or 0), 1)}%. "
        f"В спокойном феврале 2024 эти же величины {_num(cases['quiet_2024_02_mean_abs_z'], 2)} и "
        f"{_num(100 * (cases['share_abs_z_above_2_february'] or 0), 1)}%."
    )
    lines.append("")
    lines.extend(_case_notes(cases))
    if cases["examples"]:
        lines.append("")
        lines.append("Крупнейшие отклонения от прогноза в апреле 2024:")
        for item in cases["examples"]:
            title = item["name"] or f"территория {item['territory_id']}"
            region = item["region"] or "регион не сопоставлен"
            lines.append(f"- {title} ({region}), |z| = {_num(item['abs_z_2024_04'], 2)}")
    return "\n".join(lines)


def _mae_at(by_h: pd.DataFrame, model: str, horizon: int):
    part = by_h.loc[(by_h["model"] == model) & (by_h["horizon"] == horizon), "mae"]
    if part.empty:
        return None
    return float(part.iloc[0])


def _forecast_reading(by_h: pd.DataFrame, overall: pd.DataFrame) -> str:
    lines = []
    panel_12 = _mae_at(by_h, "panel_seasonal", 12)
    naive_12 = _mae_at(by_h, "seasonal_naive", 12)
    if panel_12 is not None and naive_12 is not None and abs(panel_12 - naive_12) < 1e-6:
        lines.append(
            "На горизонте 12 месяцев единственное отсечение — декабрь 2023. "
            "Общий рост за предыдущий год посчитать нельзя: 2022 года в данных нет. "
            "Поэтому панельный прогноз на этом горизонте совпадает с сезонным наивным."
        )
    light = overall.loc[overall["model"] == "lightgbm"]
    if not light.empty and int(light["n_horizons"].iloc[0]) < 4:
        bits = []
        for horizon in (1, 3, 6):
            left = _mae_at(by_h, "panel_seasonal", horizon)
            right = _mae_at(by_h, "lightgbm", horizon)
            if left is None or right is None:
                continue
            bits.append(f"{horizon} мес. {_num(left, 0)} против {_num(right, 0)}")
        lines.append(
            "Средняя MAE LightGBM усредняет только горизонты 1, 3 и 6: пар для обучения на 12 месяцев нет. "
            "Сравнивать это число со средней по четырём горизонтам нельзя. "
            "На общих горизонтах панельный прогноз и LightGBM дают MAE " + "; ".join(bits) + "."
        )
    lines.append(
        "Chronos-Bolt смотрит на каждый ряд отдельно и не видит соседей. "
        "На коротком горизонте он близок к сильным моделям, на 12 месяцах ошибка резко растёт: "
        "одного сезонного цикла модели не хватает."
    )
    return "\n\n".join(lines)


def _detector_reading(methods: dict, winner: str) -> str:
    lines = [
        f"Лучший метод на этой проверке: {DETECTOR_LABEL.get(winner, winner)}.",
        "Синтетический скачок — это рост уровня на 30%. Панельный z-score как раз сравнивает муниципалитет с остальными в том же месяце, поэтому на таком скачке он сильный. Это не значит, что любой реальный шок будет таким же заметным.",
    ]
    hybrid = methods.get("panel_news") or {}
    if hybrid.get("alpha") == 0:
        lines.append(
            "Вес региональной новости, который лучше всего прошёл синтетику, равен нулю. "
            "Гибрид не обогнал статистический детектор: искусственный скачок в календарь событий не записан. "
            "Новости остаются признаком прогноза и способом разобрать реальные эпизоды, а не источником более ранней тревоги на этой проверке."
        )
    missed = [DETECTOR_LABEL.get(name, name) for name, item in methods.items() if item.get("within_fpr_budget") is False]
    if missed:
        lines.append(
            "Не уложились в долю ложных тревог около 10%: " + ", ".join(missed) + ". "
            "Для них в таблице фактическая доля при самом спокойном из перебранных порогов."
        )
        if any(name.startswith("pelt") and methods[name].get("within_fpr_budget") is False for name in methods):
            lines.append(
                "PELT на ряде из 24 точек путает декабрьский пик со сменой режима, поэтому удержать ложные тревоги около 10% не получается."
            )
    return "\n\n".join(lines)


def _skill_markdown(by_h: pd.DataFrame, overall: pd.DataFrame) -> str:
    lines = [
        "Дополнительно: sMAPE и skill score `1 − MAE модели / MAE Prophet` на тех же горизонтах. "
        "Положительный skill значит, что модель точнее базы Prophet. "
        "У LightGBM skill посчитан по горизонтам 1, 3 и 6, потому что горизонта 12 месяцев у него нет."
    ]
    for _, item in overall.sort_values("mae").iterrows():
        model = item["model"]
        paired_model = []
        paired_prophet = []
        for horizon in sorted(by_h.loc[by_h["model"] == model, "horizon"]):
            left = _mae_at(by_h, model, int(horizon))
            right = _mae_at(by_h, "prophet", int(horizon))
            if left is None or right is None:
                continue
            paired_model.append(left)
            paired_prophet.append(right)
        if not paired_prophet or not sum(paired_prophet):
            continue
        skill = 1 - (sum(paired_model) / len(paired_model)) / (sum(paired_prophet) / len(paired_prophet))
        lines.append(
            f"- {MODEL_LABEL.get(model, model)}: sMAPE {_num(100 * float(item['smape']), 1)}%, "
            f"skill {_num(skill, 2)}"
        )
    return "\n".join(lines)


def _html(profile, metrics, best, detection, chronos_note, prophet_note: str = "") -> str:
    by_h = metrics.loc[metrics["horizon"] > 0]
    overall = metrics.loc[metrics["horizon"] == 0].sort_values("mae")
    prophet_mae = float(overall.loc[overall["model"] == "prophet", "mae"].iloc[0])
    best_mae = float(overall.loc[overall["model"] == best, "mae"].iloc[0])
    winner = detection["best_detector"]
    rows = []
    for _, item in overall.iterrows():
        cells = []
        for horizon in (1, 3, 6, 12):
            part = by_h.loc[(by_h["model"] == item["model"]) & (by_h["horizon"] == horizon)]
            if part.empty:
                cells.append("—")
            else:
                cells.append(
                    f"{_num(float(part['mae'].iloc[0]), 0)}"
                    f"<br><span class=\"sub\">{_num(float(part['r2'].iloc[0]), 2)}</span>"
                )
        mark = "best" if item["model"] == best else ""
        rows.append(
            f"<tr class=\"{mark}\">"
            f"<td>{html.escape(MODEL_LABEL.get(item['model'], item['model']))}</td>"
            + "".join(f"<td>{c}</td>" for c in cells)
            + f"<td>{_num(item['mae'], 1)}</td><td>{_num(item['r2'], 2)}</td></tr>"
        )
    det_rows = []
    for name in ["pelt_raw", "pelt_resid", "bocpd", "cusum", "panel_z", "panel_news", "news_only"]:
        item = detection["methods"].get(name)
        if not item:
            continue
        delay = "—" if item["median_delay"] is None else _num(item["median_delay"], 1)
        mark = "best" if name == winner else ""
        det_rows.append(
            f"<tr class=\"{mark}\">"
            f"<td>{html.escape(DETECTOR_LABEL.get(name, name))}</td>"
            f"<td>{_num(100 * item['fpr'], 1)}%</td>"
            f"<td>{_num(100 * item['tpr'], 1)}%</td>"
            f"<td>{_num(100 * item['precision'], 1)}%</td>"
            f"<td>{delay}</td></tr>"
        )
    charts = []
    for item in detection["cases"]["examples"][:3]:
        title = item["name"] or f"территория {item['territory_id']}"
        charts.append(
            f"<figure><figcaption>{html.escape(title)} · {html.escape(item['region'] or '')} · |z| в апреле 2024 = {_num(item['abs_z_2024_04'], 2)}</figcaption>"
            f"{_spark(item['actual'], item['forecast_2024'])}"
            "<p class=\"legend\"><span class=\"fact\">факт</span><span class=\"fit\">прогноз на месяц вперёд</span></p></figure>"
        )
    flood_bits = []
    for item in detection["cases"]["floods"]:
        if not item["n"]:
            continue
        flood_bits.append(
            f"<li>{html.escape(str(item['region']).capitalize())}, {item['month']}: |z| региона {_num(item['mean_abs_z'], 2)} "
            f"против {_num(item['outside_mean_abs_z'], 2)} у остальных ({item['n']} территорий)</li>"
        )
    case_notes = "".join(f"<p>{html.escape(text)}</p>" for text in _case_notes(detection["cases"]))
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Расходы муниципалитетов: прогноз и шоки</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; background: #f3f0e8; color: #1c1915; font: 18px/1.55 Georgia, "Iowan Old Style", serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 80px; }}
    h1 {{ font-weight: 500; font-size: 40px; line-height: 1.15; margin: 0 0 12px; }}
    h2 {{ font-weight: 500; font-size: 28px; margin: 0 0 8px; }}
    h3 {{ font-family: "Segoe UI", sans-serif; font-weight: 600; font-size: 18px; margin: 22px 0 8px; }}
    p, li {{ font-family: "Segoe UI", sans-serif; font-size: 16.5px; }}
    .lead {{ font-size: 20px; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 18px 0 8px; }}
    nav a {{ font-family: "Segoe UI", sans-serif; font-size: 14px; color: #0e6b45; background: white; border: 1px solid #ddd6c8; text-decoration: none; padding: 8px 12px; }}
    .slide {{ background: white; border: 1px solid #ddd6c8; padding: 28px 22px; margin: 22px 0; }}
    .kicker {{ font-family: "Segoe UI", sans-serif; font-size: 13px; color: #0e6b45; margin: 0 0 8px; }}
    .stat {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
    .stat div {{ background: #f7f4ec; border: 1px solid #e4dccb; padding: 12px 16px; min-width: 140px; }}
    .stat b {{ display: block; font-family: "Segoe UI", sans-serif; font-size: 22px; }}
    .stat span {{ font-family: "Segoe UI", sans-serif; font-size: 13px; color: #5c564c; }}
    .flow {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; margin: 16px 0; }}
    .flow div {{ background: #f7f4ec; border: 1px solid #e4dccb; padding: 12px 14px; flex: 1 1 140px; font-family: "Segoe UI", sans-serif; font-size: 15px; }}
    .flow b {{ display: block; font-size: 13px; color: #0e6b45; font-weight: 600; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e4dccb; padding: 14px; background: #f7f4ec; }}
    .card b {{ display: block; font-family: "Segoe UI", sans-serif; margin-bottom: 6px; }}
    .scroll {{ overflow-x: auto; }}
    table {{ width: 100%; min-width: 640px; border-collapse: collapse; font-family: "Segoe UI", sans-serif; font-size: 14px; }}
    th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #e4dccb; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ font-weight: 600; }}
    tr.best td {{ background: #e7f5ee; font-weight: 600; }}
    figure {{ margin: 18px 0; border: 1px solid #e4dccb; padding: 12px; }}
    figcaption {{ font-family: "Segoe UI", sans-serif; font-size: 14px; margin-bottom: 6px; }}
    svg {{ width: 100%; height: auto; }}
    .legend {{ display: flex; gap: 16px; font-size: 13px; }}
    .fact::before, .fit::before {{ content: ""; display: inline-block; width: 22px; height: 0; border-top: 2px solid #1c1915; margin-right: 6px; vertical-align: middle; }}
    .fit::before {{ border-top-style: dashed; border-top-color: #0e6b45; }}
    .note, .sub {{ color: #5c564c; }}
    .sub {{ font-size: 12px; }}
    ol {{ margin: 8px 0 0; padding-left: 22px; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
<main>
  <p class="kicker">Презентация итогов · задача 2 конкурса СберИндекса</p>
  <h1>Прогноз расходов муниципалитетов и поиск структурных сдвигов</h1>
  <p class="lead">Лучший прогноз — {html.escape(MODEL_LABEL.get(best, best))}: средняя MAE {_num(best_mae, 1)} против {_num(prophet_mae, 1)} у Prophet. Лучший детектор — {html.escape(DETECTOR_LABEL.get(winner, winner))}.</p>
  <nav>
    <a href="#metrics">1. Сравнение и метрики</a>
    <a href="#architecture">2. Архитектура</a>
    <a href="#cases">3. Реальные примеры</a>
  </nav>
  <div class="stat">
    <div><b>{profile['n_territories']}</b><span>муниципалитетов</span></div>
    <div><b>24</b><span>месяца, 2023–2024</span></div>
    <div><b>1, 3, 6, 12</b><span>горизонта прогноза</span></div>
    <div><b>MAE и R²</b><span>метрики оценки</span></div>
  </div>

  <section class="slide" id="metrics">
    <p class="kicker">Блок 1 · результаты сравнения и метрики</p>
    <h2>Сравнение моделей прогноза и детекторов сдвига</h2>
    <p>Главная метрика — MAE, чем меньше, тем точнее. Рядом в каждой ячейке стоит R². Зелёная строка — выбранная модель или метод. Прогноз проверялся на горизонтах 1, 3, 6 и 12 месяцев.</p>
    <h3>Модели прогнозирования</h3>
    <div class="scroll"><table>
      <thead><tr><th>Модель</th><th>1 мес.</th><th>3 мес.</th><th>6 мес.</th><th>12 мес.</th><th>Средняя MAE</th><th>Средний R²</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    <p class="note">В ячейке горизонта сверху MAE, снизу R². У LightGBM нет оценки на 12 месяцев: для обучения не хватает 2022 года, поэтому его среднюю MAE нельзя сравнивать со средней по четырём горизонтам. Skill лучшей модели относительно Prophet: {_num(1 - best_mae / prophet_mae, 2)}.</p>
    {_paragraphs(_forecast_reading(by_h, overall))}
    <h3>Методы обнаружения структурных сдвигов</h3>
    <p>Порог каждого метода подобран так, чтобы на территориях без искусственного скачка ложных тревог было около 10%. В проверке расход 80 муниципалитетов с случайного месяца умножен на 1,3.</p>
    <div class="scroll"><table>
      <thead><tr><th>Метод</th><th>Ложные тревоги</th><th>Полнота</th><th>Точность</th><th>Задержка, мес.</th></tr></thead>
      <tbody>{''.join(det_rows)}</tbody>
    </table></div>
    {_paragraphs(_detector_reading(detection["methods"], winner))}
  </section>

  <section class="slide" id="architecture">
    <p class="kicker">Блок 2 · архитектура решения</p>
    <h2>Как устроено решение</h2>
    <div class="flow">
      <div><b>1. Данные</b>Расходы, дороги и доступность рынков СберИндекса, справочник территорий, ставка, ИПЦ, календарь событий.</div>
      <div><b>2. Панель</b>Одна строка — муниципалитет и месяц. Цель — «все категории», остальное идёт в признаки.</div>
      <div><b>3. Прогноз</b>Prophet, сезонный наивный, панельный сезонный, LightGBM, Chronos-Bolt и медиана двух последних.</div>
      <div><b>4. Метрики</b>Прямой прогноз на 1, 3, 6 и 12 месяцев. Сравнение по MAE и R².</div>
      <div><b>5. Сдвиг</b>Остаток лучшего прогноза на один месяц подаётся в PELT, байесовский детектор, CUSUM, панельный z-score и новости.</div>
      <div><b>6. Вывод</b>Выбирается модель точнее Prophet и детектор с высокой полнотой при ложных тревогах около 10%.</div>
    </div>
    <p>Новость стыкуется с расходом по месяцу и региону из справочника. Решение по ставке относится ко всей стране. На дату прогноза видны только уже случившиеся события. {html.escape(chronos_note or "")}</p>
    <p class="note">{html.escape(prophet_note or "")}</p>
  </section>

  <section class="slide" id="cases">
    <p class="kicker">Блок 3 · интерпретация на реальных примерах</p>
    <h2>Что видно на событиях 2024 года</h2>
    <div class="cards">
      <div class="card"><b>Паводок, Оренбург, апрель 2024</b>Средний |z| региона не выше, чем у остальной страны. Месячный расход этот шок не показывает.</div>
      <div class="card"><b>Теракт в «Крокусе», Москва, март 2024</b>Москва без Московской области выделяется на фоне страны: это локальный сдвиг месяца, а не общий календарный пик.</div>
      <div class="card"><b>Ставка 18%, июль 2024</b>Средний |z| {_num(detection['cases']['rate_hike_2024_07_mean_abs_z'], 2)}, доля |z| &gt; 2 равна {_num(100 * (detection['cases']['share_abs_z_above_2_july'] or 0), 1)}%. В спокойном феврале {_num(detection['cases']['quiet_2024_02_mean_abs_z'], 2)} и {_num(100 * (detection['cases']['share_abs_z_above_2_february'] or 0), 1)}%. Общая для страны новость не выглядит местным шоком.</div>
    </div>
    <ul>{''.join(flood_bits) or '<li>Региональные сопоставления не собрались: проверьте поле региона в справочнике.</li>'}</ul>
    {case_notes}
    {''.join(charts)}
    <p class="note">Полный текст методологии — в report/methodology.md.</p>
  </section>
</main>
</body>
</html>
"""
