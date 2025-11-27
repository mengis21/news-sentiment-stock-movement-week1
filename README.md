# B8W1: Predicting Price Moves with News Sentiment

Applied analytics for the 10 Academy Week 1 challenge: quantify daily financial-news tone, enrich price histories with technical indicators, and benchmark how sentiment correlates with next-day returns.

## Repository layout

```
.
├── .github/workflows       # CI for lint/tests
├── .vscode                 # Local editor defaults
├── data                    # Raw + processed data assets
├── notebooks               # Jupyter notebooks (EDA, reports)
├── reports                 # Interim/final PDF exports
├── scripts                 # CLI helpers
├── src                     # Reusable Python modules
├── tests                   # Unit tests and fixtures
├── requirements.txt        # Challenge dependencies
└── README.md               # You are here
```

## Local setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place the provided CSVs under `data/raw/` (tickers + `news_headlines.csv`).
3. Verify the helper libraries by running the unit tests: `pytest`.
4. Execute the CLI workflow (details below) or the notebook in `notebooks/` for ad-hoc EDA.

## Analytics CLI

`scripts/run_interim.py` now covers the Task 2/3 rubric in one command:

```bash
.venv/bin/python scripts/run_interim.py \
  --news data/raw/news_headlines.csv \
  --price-dir data/raw \
  --tickers AAPL AMZN GOOG META MSFT NVDA \
  --topics 4 --topic-words 8 --tfidf-top-k 40 \
  --max-news-rows 20000 --sentiment-shift-days 1 \
  --output reports/interim_run.json
```

Key behaviors:

- Combines TextBlob + VADER sentiment, aggregates per ticker/day, and aligns with daily returns (including optional lag via `--sentiment-shift-days`).
- Builds MA/RSI/MACD/Bollinger/volatility indicators with optional PyNance overlays; use `--disable-pynance` to skip the PyNance path if it slows the run.
- Extracts TF-IDF keywords and LDA topics; `--skip-tfidf`/`--skip-topics` provide faster iterations. `--max-news-rows` caps headline volume for quick dry runs.
- Emits a JSON summary (or prints to stdout) containing publisher stats, sentiment aggregates, indicator snapshots, and correlation tables ready for reporting.

The latest run (capped at 20k headlines for turnaround) lives in `reports/interim_run.json`.

## Interim goals

| Task                      | Status  | Notes                                            |
| ------------------------- | ------- | ------------------------------------------------ |
| Git + repo structure      | ✅ Done | folders, CI, env files                           |
| Exploratory data analysis | ✅ Done | `01_interim_analysis.ipynb`                      |
| Technical indicators      | ✅ Done | MA, RSI, MACD, Bollinger, volatility             |
| Sentiment + correlation   | ✅ Done | TextBlob + VADER with ticker-level lag alignment |
| Documentation             | ✅ Done | README + in-repo interim report                  |

## References

- Challenge brief (10 Academy Week 1).
- [TextBlob docs](https://textblob.readthedocs.io/en/dev/)
- [TA library](https://github.com/bukosabino/ta)
- [Investopedia](https://www.investopedia.com/)
