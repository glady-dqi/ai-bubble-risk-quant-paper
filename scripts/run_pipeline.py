import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import sadf, gsadf, rolling_adf, bootstrap_adf_crit
from src.lppl import fit_lppl
from src.predict import build_features, build_labels, walk_forward_prob

prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
ai = pd.read_csv('data/ai_vs_spy.csv', index_col=0, parse_dates=True)

ai_basket = ai['AI_BASKET']
spy = ai['SPY']

# Descriptive stats
ret_ai = ai_basket.pct_change().dropna()
ret_spy = spy.pct_change().dropna()

def stats(s):
    return pd.Series({
        'mean_daily': s.mean(),
        'vol_daily': s.std(),
        'mean_ann': s.mean()*252,
        'vol_ann': s.std()*np.sqrt(252),
        'max_dd': (s.add(1).cumprod()/s.add(1).cumprod().cummax()-1).min()
    })

stats_tbl = pd.concat([
    stats(ret_ai).rename('AI Basket'),
    stats(ret_spy).rename('SPY')
], axis=1)
stats_tbl.index = ['mean daily','vol daily','mean ann','vol ann','max dd']
stats_tbl.to_csv('results/table_descriptive.csv')

# Breadth proxy: share of AI tickers above 200d MA
ai_prices = prices[[c for c in prices.columns if c in ['NVDA','MSFT','GOOGL','AMZN','META','AAPL','TSLA','AMD','AVGO','ASML','SMH']]].dropna()
ma200 = ai_prices.rolling(200).mean()
breadth = (ai_prices > ma200).mean(axis=1)
breadth.to_csv('results/breadth.csv')

# Concentration proxy: price-weighted HHI
weights = ai_prices.div(ai_prices.sum(axis=1), axis=0)
hhi = (weights**2).sum(axis=1)
hhi.to_csv('results/hhi.csv')

# SADF/GSADF with bootstrap critical
sadf_stat = sadf(ai_basket, step=5)
gsadf_stat = gsadf(ai_basket.tail(800), step=25)
crit = bootstrap_adf_crit(window=200, sims=300, alpha=0.95)

adf_roll = rolling_adf(ai_basket, window=200)

# Bubble dating: ADF above critical
bubble = adf_roll > crit

plt.figure(figsize=(8,4))
ax = plt.gca()
ax.plot(ai_basket.index, ai_basket.values, label='AI Basket')
ax.fill_between(ai_basket.index, ai_basket.min(), ai_basket.max(), where=bubble.reindex(ai_basket.index, method='ffill').fillna(False), color='red', alpha=0.1, label='Explosive episodes')
ax.set_title('AI Basket with GSADF-style explosive episodes')
ax.legend()
plt.tight_layout()
plt.savefig('figures/gsadf_bubble_overlay.png', dpi=150)

plt.figure()
adf_roll.plot()
plt.axhline(crit, color='r', linestyle='--', label='95% critical')
plt.title('Rolling ADF (200-day)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/rolling_adf.png', dpi=150)

# LPPL sensitivity
window = 500
params = fit_lppl(ai_basket.tail(window))

# rolling LPPL tc distribution
tc_list = []
for i in range(0, 200, 20):
    series = ai_basket.iloc[-(window+i):-i if i>0 else None]
    try:
        p = fit_lppl(series)
        tc_list.append(p[3])
    except Exception:
        pass

tc_series = pd.Series(tc_list)
tc_series.to_csv('results/lppl_tc_samples.csv')
tc_q = tc_series.quantile([0.1,0.5,0.9]).to_dict()

plt.figure()
plt.hist(tc_list, bins=15)
plt.title('LPPL critical time distribution (rolling fits)')
plt.tight_layout()
plt.savefig('figures/lppl_tc_hist.png', dpi=150)

# Prob model (3m only) + empirical baselines (6m/12m)
feat = build_features(ai_basket, spy)

labels_3m = build_labels(ai_basket, horizon=63)
probs_3m = walk_forward_prob(feat, labels_3m, split_date='2022-01-01')

# align labels
lab = labels_3m.reindex(probs_3m.index).dropna()
probs_3m = probs_3m.reindex(lab.index)

brier = brier_score_loss(lab, probs_3m)
auc = roc_auc_score(lab, probs_3m)
base_3m = lab.mean()

# empirical baselines for 6m/12m
labels_6m = build_labels(ai_basket, horizon=126)
labels_12m = build_labels(ai_basket, horizon=252)
base_6m = labels_6m.mean()
base_12m = labels_12m.mean()

metrics_df = pd.DataFrame([
    ['3m', brier, auc, base_3m, probs_3m.mean(), probs_3m.iloc[-1], 'logit'],
    ['6m', None, None, base_6m, None, None, 'empirical'],
    ['12m', None, None, base_12m, None, None, 'empirical'],
], columns=['Horizon','Brier','AUC','Base rate','Prob mean','Prob last','Method'])
metrics_df.to_csv('results/table_crash_probs.csv', index=False)

# calibration curve (3m)
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(lab, probs_3m, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o')
plt.plot([0,1],[0,1],'--')
plt.title('Calibration curve (3m)')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.tight_layout()
plt.savefig('figures/calibration_3m.png', dpi=150)

# Save probs
probs_3m.to_csv('results/crash_prob_3m.csv')

# Robustness
# Alternative universes
semi = ['NVDA','AMD','AVGO','ASML','SMH']
ex_tsla = [t for t in ai_prices.columns if t != 'TSLA']

rob_rows = []
for name, basket in [('baseline', ai_prices), ('semis', ai_prices[semi]), ('no tsla', ai_prices[ex_tsla])]:
    b = basket.pct_change().mean(axis=1).add(1).cumprod().dropna()
    s = sadf(b, step=5)
    labels15 = build_labels(b, horizon=63, drawdown=0.15).mean()
    labels20 = build_labels(b, horizon=63, drawdown=0.20).mean()
    labels30 = build_labels(b, horizon=63, drawdown=0.30).mean()
    pre = b.loc[:'2019-12-31']
    post = b.loc['2020-01-01':]
    s_pre = sadf(pre, step=5)
    s_post = sadf(post, step=5)
    rob_rows.append([name, s, s_pre, s_post, labels15, labels20, labels30])

rob_df = pd.DataFrame(rob_rows, columns=['Universe','SADF','SADF pre-2020','SADF post-2020','DD15 rate','DD20 rate','DD30 rate'])
rob_df.to_csv('results/robustness.csv', index=False)
rob_df.to_latex('results/table_robustness.tex', index=False, float_format="%.4f")

# Save stats
with open('results/stats.txt','w') as f:
    f.write(f"SADF: {sadf_stat}\n")
    f.write(f"GSADF (sub-sampled): {gsadf_stat}\n")
    f.write(f"ADF_crit_95: {crit}\n")
    f.write(f"CrashProb_3m_mean: {probs_3m.mean()}\n")
    f.write(f"CrashProb_3m_last: {probs_3m.iloc[-1]}\n")
    f.write(f"CrashProb_6m_empirical: {base_6m}\n")
    f.write(f"CrashProb_12m_empirical: {base_12m}\n")
    f.write(f"Brier_3m: {brier}\n")
    f.write(f"AUC_3m: {auc}\n")
    f.write(f"LPPL_tc_q10: {tc_q.get(0.1)}\n")
    f.write(f"LPPL_tc_q50: {tc_q.get(0.5)}\n")
    f.write(f"LPPL_tc_q90: {tc_q.get(0.9)}\n")

# LaTeX tables
stats_tbl.to_latex('results/table_descriptive.tex', float_format="%.4f")
metrics_df.to_latex('results/table_crash_probs.tex', index=False, float_format="%.4f")

print('done')
