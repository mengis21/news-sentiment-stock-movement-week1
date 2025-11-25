# B8W1: Predicting Price Moves with News Sentiment

Late interim submission for the 10 Academy Week 1 challenge. The focus is to connect daily financial news sentiment with stock price moves through reproducible Python workflows.

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
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Drop the provided CSV files into `data/raw/`.
3. Run the interim notebook under `notebooks/` for the EDA flow.

## Interim goals

| Task                      | Status  | Notes                                 |
| ------------------------- | ------- | ------------------------------------- |
| Git + repo structure      | ✅ Done | folders, CI, env files                |
| Exploratory data analysis | ✅ Done | `01_interim_analysis.ipynb`           |
| Technical indicators      | ✅ Done | moving average + RSI helpers          |
| Sentiment + correlation   | ✅ Done | TextBlob baseline, align with returns |
| Documentation             | ✅ Done | README + interim write-up (external)  |

## References

- Challenge brief (10 Academy Week 1).
- [TextBlob docs](https://textblob.readthedocs.io/en/dev/)
- [TA library](https://github.com/bukosabino/ta)
- [Investopedia](https://www.investopedia.com/)
