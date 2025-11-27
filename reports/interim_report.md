# B8W1 Interim Report

_Date: 27 Nov 2025 (submitted late)_

## Business Objective

Nova Financial Solutions wants to use daily financial headlines to anticipate stock price swings. The immediate focus is to quantify news sentiment for each ticker, align it with same-day or next-day price returns, and study the correlation strength. The findings will guide simple trading signals (for example, fade negative news or ride strong positive clusters) once the pipeline is validated on the full Financial News and Stock Price Integration Dataset (FNSPID).

## Completed Work and Key Findings

- **Production-ready loaders**: `src/data_io` handles both single-CSV and directory-based OHLCV ingestion along with ticker/date filters. News ingestion normalizes timestamps and publisher names for reliable grouping.
- **Expanded feature engineering**: Added MA/RSI/MACD/Bollinger/volatility indicator helpers (with PyNance overlays when available). `src/sentiment` now supports TextBlob + VADER scores, plus ticker/day aggregation for multiple sentiment metrics.
- **Unified CLI workflow**: `scripts/run_interim.py` orchestrates the full pipeline—multi-ticker prices, sentiment enrichment, TF-IDF keywords, LDA topics, PyNance metrics (optional), and ticker-level correlation tables. The latest run (20k-headline cap for turnaround) is stored in `reports/interim_run.json`.
- **Exploratory highlights** (from the latest run):
  - Headline lengths average **74 characters** with a long tail up to 397 chars.
  - Coverage is dominated by Benzinga contributors; the top three bylines (Paul Quintaro, Lisa Levin, Benzinga Newsdesk) account for **~41%** of the capped sample.
  - Publishing cadence peaks between **11:00–14:00 UTC**, hinting at midday U.S. market focus.
  - TF-IDF keywords surface expected finance terms ("stocks", "earnings", "price target") plus ticker-specific narratives ("Alcoa", "upgrades/downgrades").
  - Average per-ticker correlations between daily returns and the aggregated VADER/Blob metrics hover around **0.33** when sentiment is shifted by one day, signalling moderate predictive power worth deeper validation.
- **Notebook + tests**: `notebooks/01_interim_analysis.ipynb` mirrors the report narrative, while `pytest` covers the loaders, sentiment helpers, advanced indicators, and correlation utilities to guard regressions.

## Next Steps and Key Areas of Focus

1. **Scale beyond capped sample**: The CLI supports `--max-news-rows`; schedule a full historical run (no cap, PyNance enabled) once runtime constraints ease.
2. **Model experimentation**: Evaluate FinBERT or other finance-specific transformers to complement TextBlob/VADER and compare correlation lift.
3. **Signal validation**: Stress-test correlations with rolling windows + cross-validation, then translate promising combos into backtest-ready rules.
4. **Final deliverable**: Transform this markdown + JSON outputs into the final PDF with charts (sentiment timelines, indicator overlays) and explicit commentary on lag structures.

## Structure, Clarity, and Style

- Notebook `notebooks/01_interim_analysis.ipynb` follows the same flow as this write-up for quick traceability.
- Scripts stay dependency-light (`pandas`, `ta`, `pynance`, `nltk`, `sklearn`) so reviewers can rerun everything with `pip install -r requirements.txt`.
- Comments were kept to intent-only hints; business context lives in this document instead of the code.

## Blockers and Risks

- **PyNance stability**: Some tickers lack enough history for PyNance growth metrics, so the CLI downgrades gracefully but longer histories would improve coverage.
- **Volume vs. runtime**: Full 1.4M-headline runs take significant time; the `--max-news-rows` cap is a stopgap until more compute is available.
- **Late submission**: Still noted; mitigation is tight version control, clear documentation, and reproducible scripts so reviewers can audit progress quickly.
