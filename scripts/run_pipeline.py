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
from src.factors import build_factors, residualize

prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
baskets = pd.read_csv('data/baskets.csv', index_col=0, parse_dates=True)

ai_basket = baskets['AI_BASKET']
nonai_tech = baskets['NONAI_TECH']
ai_semi = baskets['AI_SEMI']
nonai_semi = baskets['NONAI_SEMI']
spy = baskets['SPY']
xlk = baskets['XLK']

# descriptive stats
ret_ai = ai_basket.pct_change().dropna()
ret_spy = spy.pct_change().dropna()

def stats(s):
    return pd.Series({
        'mean daily': s.mean(),
        'vol daily': s.std(),
        'mean ann': s.mean()*252,
        'vol ann': s.std()*np.sqrt(252),
        'max dd': (s.add(1).cumprod()/s.add(1).cumprod().cummax()-1).min()
    })

stats_tbl = pd.concat([
    stats(ret_ai).rename('AI Basket'),
    stats(ret_spy).rename('SPY')
], axis=1)
stats_tbl.to_csv('results/table_descriptive.csv')

# breadth & concentration
AI_TICKERS = ['NVDA','MSFT','GOOGL','AMZN','META','AAPL','TSLA','AMD','AVGO','ASML','SMH']
NONAI_TECH = ['IBM','ORCL','CSCO','INTC','TXN','QCOM','ADBE']
AI_SEMI = ['NVDA','AMD','AVGO','ASML','SMH']
NONAI_SEMI = ['INTC','TXN','QCOM','MU','NXPI']

ai_prices = prices[AI_TICKERS].dropna()
nonai_prices = prices[NONAI_TECH].dropna()
ai_semi_prices = prices[AI_SEMI].dropna()
nonai_semi_prices = prices[NONAI_SEMI].dropna()


def breadth_disp_hhi(px):
    ma200 = px.rolling(200).mean()
    breadth = (px > ma200).mean(axis=1)
    disp = px.pct_change().std(axis=1)
    rets = px.pct_change().dropna()
    weights = rets.abs().div(rets.abs().sum(axis=1), axis=0)
    hhi_ret = (weights**2).sum(axis=1)
    return breadth, disp, hhi_ret

breadth, disp, hhi_ret = breadth_disp_hhi(ai_prices)
non_b, non_d, non_h = breadth_disp_hhi(nonai_prices)
ase_b, ase_d, ase_h = breadth_disp_hhi(ai_semi_prices)
na_b, na_d, na_h = breadth_disp_hhi(nonai_semi_prices)

breadth.to_csv('results/breadth.csv')
disp.to_csv('results/dispersion.csv')
hhi_ret.to_csv('results/hhi_return.csv')

summary = pd.DataFrame([
    ['AI basket', breadth.mean(), breadth.iloc[-1], disp.mean(), disp.iloc[-1], hhi_ret.mean(), hhi_ret.iloc[-1]],
    ['Non-AI tech', non_b.mean(), non_b.iloc[-1], non_d.mean(), non_d.iloc[-1], non_h.mean(), non_h.iloc[-1]],
    ['AI semis', ase_b.mean(), ase_b.iloc[-1], ase_d.mean(), ase_d.iloc[-1], ase_h.mean(), ase_h.iloc[-1]],
    ['Non-AI semis', na_b.mean(), na_b.iloc[-1], na_d.mean(), na_d.iloc[-1], na_h.mean(), na_h.iloc[-1]],
], columns=['Universe','Breadth mean','Breadth last','Dispersion mean','Dispersion last','HHI mean','HHI last'])
summary.to_csv('results/table_concentration.csv', index=False)
summary.to_latex('results/table_concentration.tex', index=False, float_format="%.4f")

# factors and residuals
factors = build_factors(prices)
res_ai, model_ai = residualize(ai_basket, factors)
res_nonai, _ = residualize(nonai_tech, factors)
res_ai_semi, _ = residualize(ai_semi, factors)
res_nonai_semi, _ = residualize(nonai_semi, factors)

# SADF/GSADF
sadf_ai = sadf(ai_basket, step=5)
sgsadf_ai = gsadf(ai_basket.tail(800), step=25)
crit = bootstrap_adf_crit(window=200, sims=300, alpha=0.95)

# residual diagnostics
sadf_ai_res = sadf(res_ai, step=5)
sgsadf_ai_res = gsadf(res_ai.tail(800), step=25)

# controls
sadf_nonai = sadf(nonai_tech, step=5)
sgsadf_nonai = gsadf(nonai_tech.tail(800), step=25)

sadf_ai_semi = sadf(ai_semi, step=5)
sgsadf_ai_semi = gsadf(ai_semi.tail(800), step=25)

sadf_nonai_semi = sadf(nonai_semi, step=5)
sgsadf_nonai_semi = gsadf(nonai_semi.tail(800), step=25)

# residual controls
sadf_nonai_res = sadf(res_nonai, step=5)
sgsadf_nonai_res = gsadf(res_nonai.tail(800), step=25)

sadf_ai_semi_res = sadf(res_ai_semi, step=5)
sgsadf_ai_semi_res = gsadf(res_ai_semi.tail(800), step=25)

sadf_nonai_semi_res = sadf(res_nonai_semi, step=5)
sgsadf_nonai_semi_res = gsadf(res_nonai_semi.tail(800), step=25)

# rolling adf + bubble overlay
adf_roll = rolling_adf(ai_basket, window=200)
bubble = adf_roll > crit

# episode counts and fraction flagged

def count_episodes(flag_series):
    flag = flag_series.fillna(False).astype(int)
    return ((flag.diff() == 1).sum())

fraction_ai = bubble.mean()
episodes_ai = count_episodes(bubble)

# residual bubble
adf_ai_res = rolling_adf(res_ai, window=200)
bubble_ai_res = adf_ai_res > crit
fraction_ai_res = bubble_ai_res.mean()
episodes_ai_res = count_episodes(bubble_ai_res)

# controls raw/residual
adf_nonai = rolling_adf(nonai_tech, window=200)
bubble_nonai = adf_nonai > crit
fraction_nonai = bubble_nonai.mean()
episodes_nonai = count_episodes(bubble_nonai)

adf_nonai_res = rolling_adf(res_nonai, window=200)
bubble_nonai_res = adf_nonai_res > crit
fraction_nonai_res = bubble_nonai_res.mean()
episodes_nonai_res = count_episodes(bubble_nonai_res)

# table: explosive dynamics
exp_table = pd.DataFrame([
    ['AI basket (raw)', sgsadf_ai, episodes_ai, fraction_ai],
    ['AI basket (residual)', sgsadf_ai_res, episodes_ai_res, fraction_ai_res],
    ['Non-AI tech (raw)', sgsadf_nonai, episodes_nonai, fraction_nonai],
    ['Non-AI tech (residual)', sgsadf_nonai_res, episodes_nonai_res, fraction_nonai_res],
], columns=['Series','GSADF','Episodes','Fraction flagged'])
exp_table.to_csv('results/table_explosive.csv', index=False)
exp_table.to_latex('results/table_explosive.tex', index=False, float_format="%.4f")

# main figure: AI vs control with explosive episodes
plt.figure(figsize=(8,4))
ax = plt.gca()
ax.plot(ai_basket.index, ai_basket/ai_basket.iloc[0], label='AI Basket')
ax.plot(nonai_tech.index, nonai_tech/nonai_tech.iloc[0], label='Non-AI Tech')
ax.fill_between(ai_basket.index, 0, (ai_basket/ai_basket.iloc[0]).max(), where=bubble.reindex(ai_basket.index, method='ffill').fillna(False), color='red', alpha=0.1, label='Explosive episodes (AI)')
ax.set_title('AI vs Non-AI Tech with Explosive Episodes')
ax.legend()
plt.tight_layout()
plt.savefig('figures/explosive_ai_vs_control.png', dpi=150)

# additional overlay (appendix)
plt.figure(figsize=(8,4))
plt.plot(ai_basket.index, ai_basket/ai_basket.iloc[0], label='AI Basket')
plt.plot(nonai_tech.index, nonai_tech/nonai_tech.iloc[0], label='Non-AI Tech')
plt.plot(ai_semi.index, ai_semi/ai_semi.iloc[0], label='AI Semis')
plt.plot(nonai_semi.index, nonai_semi/nonai_semi.iloc[0], label='Non-AI Semis')
plt.title('AI vs Controls (Indexed)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ai_vs_controls.png', dpi=150)

# LPPL sensitivity (appendix)
window = 500
params = fit_lppl(ai_basket.tail(window))

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

# Prob model (3m only)
feat = build_features(ai_basket, spy)
labels_3m = build_labels(ai_basket, horizon=63)
probs_3m = walk_forward_prob(feat, labels_3m, split_date='2022-01-01')

lab = labels_3m.reindex(probs_3m.index).dropna()
probs_3m = probs_3m.reindex(lab.index)

brier = brier_score_loss(lab, probs_3m)
auc = roc_auc_score(lab, probs_3m)
base_3m = lab.mean()

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

frac_pos, mean_pred = calibration_curve(lab, probs_3m, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o')
plt.plot([0,1],[0,1],'--')
plt.title('Calibration curve (3m)')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.tight_layout()
plt.savefig('figures/calibration_3m.png', dpi=150)

probs_3m.to_csv('results/crash_prob_3m.csv')

# Tables
stats_tbl.to_latex('results/table_descriptive.tex', float_format="%.4f")
metrics_df.to_latex('results/table_crash_probs.tex', index=False, float_format="%.4f")

# GSADF comparison table
comp = pd.DataFrame([
    ['AI raw', sadf_ai, sgsadf_ai],
    ['AI residual', sadf_ai_res, sgsadf_ai_res],
    ['Non-AI tech raw', sadf_nonai, sgsadf_nonai],
    ['Non-AI tech residual', sadf_nonai_res, sgsadf_nonai_res],
    ['AI semis raw', sadf_ai_semi, sgsadf_ai_semi],
    ['AI semis residual', sadf_ai_semi_res, sgsadf_ai_semi_res],
    ['Non-AI semis raw', sadf_nonai_semi, sgsadf_nonai_semi],
    ['Non-AI semis residual', sadf_nonai_semi_res, sgsadf_nonai_semi_res],
], columns=['Series','SADF','GSADF'])
comp.to_csv('results/table_gsadf_compare.csv', index=False)
comp.to_latex('results/table_gsadf_compare.tex', index=False, float_format="%.4f")

# Robustness
rob_rows = []
for name, basket in [('baseline', ai_prices), ('semis', ai_prices[['NVDA','AMD','AVGO','ASML','SMH']]), ('no tsla', ai_prices[[c for c in ai_prices.columns if c!='TSLA']])]:
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
    f.write(f"SADF_AI: {sadf_ai}\n")
    f.write(f"GSADF_AI: {sgsadf_ai}\n")
    f.write(f"SADF_AI_resid: {sadf_ai_res}\n")
    f.write(f"GSADF_AI_resid: {sgsadf_ai_res}\n")
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

print('done')
