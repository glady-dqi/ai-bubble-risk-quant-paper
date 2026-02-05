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

# Prob model
feat = build_features(ai_basket, spy)
labels = build_labels(ai_basket)
probs = walk_forward_prob(feat, labels, split_date='2022-01-01')
plt.figure()
probs.plot()
plt.title('Crash Probability (3m, 20% DD)')
plt.savefig('figures/crash_prob.png', dpi=150)

pd.Series(params, index=['A','B','C','tc','m','w','phi']).to_csv('results/lppl_params.csv')

with open('results/stats.txt','w') as f:
    f.write(f"SADF: {sadf_stat}\n")
    f.write(f"GSADF (approx): {gsadf_stat}\n")

print('done')
