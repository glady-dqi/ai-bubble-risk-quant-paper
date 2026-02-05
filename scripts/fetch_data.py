import yfinance as yf
import pandas as pd

AI_TICKERS = ["NVDA","MSFT","GOOGL","AMZN","META","AAPL","TSLA","AMD","AVGO","ASML","SMH"]
NONAI_TECH = ["IBM","ORCL","CSCO","INTC","TXN","QCOM","ADBE"]
AI_SEMI = ["NVDA","AMD","AVGO","ASML","SMH"]
NONAI_SEMI = ["INTC","TXN","QCOM","MU","NXPI"]
BENCH_TICKERS = ["SPY","QQQ","XLK"]
FACTORS = ["^TNX","^VIX"]

ALL = sorted(set(AI_TICKERS + NONAI_TECH + AI_SEMI + NONAI_SEMI + BENCH_TICKERS + FACTORS))

start = "2015-01-01"

data = yf.download(ALL, start=start, auto_adjust=True, progress=False)["Close"]
data.to_csv("data/prices.csv")

# build equal-weight baskets
ai = data[AI_TICKERS].dropna(how="all")
nonai = data[NONAI_TECH].dropna(how="all")
ai_semi = data[AI_SEMI].dropna(how="all")
nonai_semi = data[NONAI_SEMI].dropna(how="all")

baskets = pd.DataFrame({
    "AI_BASKET": ai.pct_change().mean(axis=1).add(1).cumprod(),
    "NONAI_TECH": nonai.pct_change().mean(axis=1).add(1).cumprod(),
    "AI_SEMI": ai_semi.pct_change().mean(axis=1).add(1).cumprod(),
    "NONAI_SEMI": nonai_semi.pct_change().mean(axis=1).add(1).cumprod(),
    "SPY": data["SPY"],
    "QQQ": data["QQQ"],
    "XLK": data["XLK"],
})

baskets.dropna().to_csv("data/baskets.csv")
print("ok", baskets.tail())
