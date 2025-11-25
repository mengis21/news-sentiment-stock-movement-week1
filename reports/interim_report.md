# B8W1 Interim Report

_Date: 25 Nov 2025 (submitted late)_

## Business Objective

Nova Financial Solutions wants to use daily financial headlines to anticipate stock price swings. The immediate focus is to quantify news sentiment for each ticker, align it with same-day or next-day price returns, and study the correlation strength. The findings will guide simple trading signals (for example, fade negative news or ride strong positive clusters) once the pipeline is validated on the full Financial News and Stock Price Integration Dataset (FNSPID).

## Completed Work and Initial Analysis

- **Environment + Git hygiene**: Repo skeleton, CI workflow, sample data, and helper scripts are in place. Work happens on branches with regular commits (late submission noted).
- **Data loading + sanity checks**: Created loaders for both news and OHLCV prices; sample CSVs confirm the schema and parsing logic.
- **Exploratory stats**: Calculated headline length distribution, publisher contribution counts, and daily article frequency (visualized in the interim notebook).
- **Sentiment baseline**: TextBlob polarity/subjectivity scores are added per headline and then averaged per calendar day for alignment with price data.
- **Technical preview**: Implemented rolling moving averages, RSI, and daily return calculations using pandas so the workflow runs even without TA-Lib.
- **Correlation dry run**: Joined daily sentiment with price returns on the sample data to confirm the math path for Task 3.

## Next Steps and Key Areas of Focus

1. **Full dataset ingestion**: Replace the sample CSVs with the full FNSPID files and batch stock histories from yfinance.
2. **Richer indicators**: Implement MACD and Bollinger Bands (either via `ta` or TA-Lib) to compare with sentiment signals.
3. **Enhanced sentiment**: Try VADER and a finance-specific transformer (FinBERT) to capture sarcasm or multi-sentence context.
4. **Correlation study**: Aggregate sentiment per ticker/day, compute lead/lag correlations with returns, and document statistically significant pairs.
5. **Reporting**: Convert this markdown into a 3-page PDF (Google Drive) with plots plus a clear note on the late submission.

## Structure, Clarity, and Style

- Notebook `notebooks/01_interim_analysis.ipynb` follows the same flow as the report for easy verification.
- Scripts + modules rely on plain Python and pandas so reviewers can rerun everything without special wheels.
- Comments stay short and only appear where the intent is not obvious.

## Blockers and Risks

- TA-Lib wheels may fail on the current machine; using the `ta` package as a fallback keeps Task 2 moving.
- Real news volume may expose timezone issues. Need to confirm UTC offsets before the final run.
- Late submission acknowledged; catching up by prioritising reproducibility and documentation.
