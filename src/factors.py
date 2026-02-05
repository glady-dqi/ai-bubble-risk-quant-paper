import pandas as pd
import numpy as np
import statsmodels.api as sm


def build_factors(prices):
    # factors: SPY, XLK, ΔTNX, ΔVIX
    df = prices[["SPY","XLK","^TNX","^VIX"]].dropna()
    rets = df[["SPY","XLK"]].pct_change()
    dtnx = df["^TNX"].diff()/100.0
    dvix = df["^VIX"].diff()/100.0
    X = pd.concat([rets, dtnx.rename("dTNX"), dvix.rename("dVIX")], axis=1).dropna()
    return X


def residualize(price_series, factors):
    r = price_series.pct_change().dropna()
    X = factors.reindex(r.index).dropna()
    r = r.reindex(X.index)
    X = sm.add_constant(X)
    model = sm.OLS(r, X).fit()
    resid = model.resid
    # build residual price index
    resid_price = resid.add(1).cumprod()
    return resid_price, model
