# Results Manifest

**Dataset source:** Yahoo Finance via yfinance (downloaded at run time)
**Sample period:** 2015-01-01 to latest available (see data/prices.csv)

## Universes
- AI basket: NVDA, MSFT, GOOGL, AMZN, META, AAPL, TSLA, AMD, AVGO, ASML, SMH
- Non-AI tech controls: IBM, ORCL, CSCO, INTC, TXN, QCOM, ADBE
- AI semiconductors: NVDA, AMD, AVGO, ASML, SMH
- Non-AI semiconductors: INTC, TXN, QCOM, MU, NXPI
- Benchmarks: SPY, QQQ, XLK

## Factor model (residualization)
$r^{AI}_t = \alpha + \beta_M r^{SPY}_t + \beta_{Tech} r^{XLK}_t + \beta_{Rates} \Delta y_t + \beta_{Vol}\Delta VIX_t + \varepsilon_t$
- Rates proxy: ^TNX (10y yield), Vol proxy: ^VIX
- Residual price index: cumulative product of $(1+\varepsilon_t)$

## Model configs
- SADF/GSADF: ADF maxlag=1, regression='c', step=5/25, min_window=50
- Rolling ADF window: 200 days
- GSADF sub-sample: last 800 observations
- LPPL constraints: m in [0.1,0.9], w in [4,15], tc in [T+1,T+200]
- Crash model: logistic regression, walk-forward (train <= 2021-12, test >= 2022-01)
- Crash event: drawdown >=20% over horizon (3/6/12 months)

## Output artifacts
- figures/gsadf_bubble_overlay.png
- figures/ai_vs_controls.png
- figures/calibration_3m.png
- figures/lppl_tc_hist.png
- results/table_descriptive.csv / .tex
- results/table_crash_probs.csv / .tex
- results/table_gsadf_compare.csv / .tex
- results/table_robustness.csv / .tex
- results/stats.txt
- results/breadth.csv, results/dispersion.csv, results/hhi_return.csv
- results/lppl_tc_samples.csv
- results/crash_prob_3m.csv
