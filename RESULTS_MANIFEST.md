# Results Manifest

**Dataset source:** Yahoo Finance via yfinance (downloaded at run time)
**Sample period:** 2015-01-01 to latest available (see data/prices.csv)
**Universe:** NVDA, MSFT, GOOGL, AMZN, META, AAPL, TSLA, AMD, AVGO, ASML, SMH
**Benchmarks:** SPY, QQQ, XLK, SOXX

## Model configs
- SADF/GSADF: ADF with maxlag=1, regression='c', step=5/25, min_window=50
- Rolling ADF window: 200 days
- GSADF sub-sample: last 800 observations
- LPPL: bounded parameters (m in [0.1,0.9], w in [4,15], tc in [T+1,T+200])
- Crash model: logistic regression, walk-forward (train <= 2021-12, test >= 2022-01)
- Crash event: drawdown >=20% over horizon (3/6/12 months)

## Output artifacts
- figures/rolling_adf.png
- figures/gsadf_bubble_overlay.png
- figures/crash_prob_3m.png
- figures/calibration_3m.png
- figures/lppl_tc_hist.png
- results/stats.txt
- results/table_descriptive.csv
- results/table_crash_probs.csv
- results/breadth.csv
- results/hhi.csv
- results/lppl_params.csv
- results/lppl_tc_samples.csv
- results/crash_prob_3m.csv
- results/crash_prob_6m.csv
- results/crash_prob_12m.csv
