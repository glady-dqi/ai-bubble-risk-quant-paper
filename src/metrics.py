import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def sadf(series, min_window=50):
    y = np.log(series.dropna())
    max_adf = -np.inf
    for end in range(min_window, len(y)):
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
