import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def sadf(series, min_window=50, step=5):
    y = np.log(series.dropna())
    max_adf = -np.inf
    for end in range(min_window, len(y), step):
        stat = adfuller(y.iloc[:end], maxlag=1, regression='c', autolag=None)[0]
        max_adf = max(max_adf, stat)
    return max_adf


def gsadf(series, min_window=50, step=20):
    y = np.log(series.dropna())
    max_adf = -np.inf
    for start in range(0, len(y)-min_window, step):
        for end in range(start+min_window, len(y), step):
            stat = adfuller(y.iloc[start:end], maxlag=1, regression='c', autolag=None)[0]
            max_adf = max(max_adf, stat)
    return max_adf


def rolling_adf(series, window=100):
    y = np.log(series.dropna())
    stats = []
    idx = []
    for i in range(window, len(y)):
        stat = adfuller(y.iloc[i-window:i], maxlag=1, regression='c', autolag=None)[0]
        stats.append(stat)
        idx.append(y.index[i])
    return pd.Series(stats, index=idx, name='adf')


def bootstrap_adf_crit(window=200, sims=300, alpha=0.95):
    # random walk bootstrap critical for ADF in rolling window
    stats = []
    for _ in range(sims):
        eps = np.random.normal(size=window)
        rw = np.cumsum(eps)
        stat = adfuller(rw, maxlag=1, regression='c', autolag=None)[0]
        stats.append(stat)
    return np.quantile(stats, alpha)
