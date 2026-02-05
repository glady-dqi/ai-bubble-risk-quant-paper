import yfinance as yf
import pandas as pd

AI_TICKERS = ["NVDA","MSFT","GOOGL","AMZN","META","AAPL","TSLA","AMD","AVGO","ASML","SMH"]
BENCH_TICKERS = ["SPY","QQQ","XLK","SOXX"]
ALL = sorted(set(AI_TICKERS + BENCH_TICKERS))

start = "2015-01-01"

data = yf.download(ALL, start=start, auto_adjust=True, progress=False)["Close"]
data.to_csv("data/prices.csv")

# build equal-weight AI basket
ai = data[AI_TICKERS].dropna(how="all")
ai_basket = ai.pct_change().mean(axis=1).add(1).cumprod()
ai_basket.name = "AI_BASKET"
bench = data["SPY"].dropna()

out = pd.concat([ai_basket, bench], axis=1).dropna()
out.to_csv("data/ai_vs_spy.csv")
print("ok", out.tail())
