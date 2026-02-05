import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_features(price, bench):
    df = pd.DataFrame({"price": price, "bench": bench}).dropna()
    r = df.pct_change().dropna()
    df = df.loc[r.index]
    df["ret_1m"] = r["price"].rolling(21).mean()
    df["ret_3m"] = r["price"].rolling(63).mean()
    df["vol_3m"] = r["price"].rolling(63).std()
    df["rel_12m"] = (df["price"]/df["bench"]).pct_change(252)
    return df.dropna()


def build_labels(price, horizon=63, drawdown=0.2):
    p = price.dropna()
    future = p.shift(-horizon)
    dd = (future - p)/p
    return (dd <= -drawdown).astype(int)


def walk_forward_prob(features, labels, split_date):
    labels = labels.reindex(features.index).dropna()
    features = features.reindex(labels.index)
    train = features.index < split_date
    X_train = features.loc[train]
    y_train = labels.loc[train]
    X_test = features.loc[~train]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    return pd.Series(probs, index=X_test.index, name='crash_prob')
