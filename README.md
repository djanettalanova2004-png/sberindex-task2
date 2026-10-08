# Прогноз расходов муниципалитетов и раннее обнаружение шоков

Решение задачи 2 конкурса СберИндекса: прогноз безналичных потребительских расходов по муниципалитетам на 1, 3, 6 и 12 месяцев и сравнение методов поиска структурных сдвигов.

Методологический отчёт и презентация уже лежат в репозитории. Их не нужно получать повторным запуском кода.

- Методологический отчёт: [report/methodology.md](report/methodology.md)
- Презентация, интерактивный лендинг: [presentation/index.html](presentation/index.html). На GitHub файл открывается как код. Готовая страница: [djanettalanova2004-png.github.io/sberindex-task2](https://djanettalanova2004-png.github.io/sberindex-task2/)

Нужен Python 3.11. На 3.14 часть библиотек (Prophet, Chronos) может не встать.

```
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.run
```

Если LightGBM не импортируется и пишет про `lib_lightgbm.dll`, поставьте [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist). На этой машине не хватало `vcomp140.dll`. Для Chronos нужен тот же пакет свежей версии: `torch` ищет `vcruntime140_threads.dll`. Если CmdStan не собирается (нет `mingw32-make`), скрипт не подменяет Prophet наивным прогнозом, а оценивает ту же спецификацию нелинейным МНК и пишет об этом в отчёте.

После прогона:

- `outputs/metrics.csv` — MAE, R² и sMAPE по моделям и горизонтам
- `outputs/detection.json` — сравнение детекторов и разбор реальных эпизодов
- `report/methodology.md` — методологический отчёт
- `presentation/index.html` — страница с результатами

Проверка на короткой выборке, чтобы убедиться, что код проходит целиком: `python -m src.run --limit 40`.

