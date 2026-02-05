import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import sadf, gsadf, rolling_adf
from src.lppl import fit_lppl
from src.predict import build_features, build_labels, walk_forward_prob

prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
ai = pd.read_csv('data/ai_vs_spy.csv', index_col=0, parse_dates=True)

ai_basket = ai['AI_BASKET']
spy = ai['SPY']

# SADF/GSADF
sadf_stat = sadf(ai_basket)
# GSADF heavy; approximate on recent window
gsadf_stat = gsadf(ai_basket.tail(800), step=25)

adf_roll = rolling_adf(ai_basket, window=200)
plt.figure()
adf_roll.plot()
plt.title('Rolling ADF (AI Basket)')
plt.axhline(-2.86, color='r', linestyle='--')
plt.savefig('figures/rolling_adf.png', dpi=150)

# LPPL
params = fit_lppl(ai_basket.tail(500))

# Prob model (3m) + empirical 6/12m
feat = build_features(ai_basket, spy)
labels_3m = build_labels(ai_basket, horizon=63)
probs_3m = walk_forward_prob(feat, labels_3m, split_date='2022-01-01')

# Empirical baseline for 6m/12m
labels_6m = build_labels(ai_basket, horizon=126)
labels_12m = build_labels(ai_basket, horizon=252)
emp_6m = labels_6m.mean()
emp_12m = labels_12m.mean()

# Plot 3m prob
plt.figure()
probs_3m.plot()
plt.title('Crash Probability (3m, 20% DD)')
plt.savefig('figures/crash_prob_3m.png', dpi=150)

# Save probs
probs_3m.to_csv('results/crash_prob_3m.csv')

pd.Series(params, index=['A','B','C','tc','m','w','phi']).to_csv('results/lppl_params.csv')

with open('results/stats.txt','w') as f:
    f.write(f"SADF: {sadf_stat}\n")
    f.write(f"GSADF (approx, sub-sampled): {gsadf_stat}\n")
    f.write(f"CrashProb_3m_mean: {probs_3m.mean()}\n")
    f.write(f"CrashProb_3m_last: {probs_3m.iloc[-1]}\n")
    f.write(f"CrashProb_6m_empirical: {emp_6m}\n")
    f.write(f"CrashProb_12m_empirical: {emp_12m}\n")

print('done')
