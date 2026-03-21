#!/usr/bin/env python3
"""
Combined Grail — DSR + Walk-Forward 自動生成
monthly_signal.yml から gen_cumulative.py の後に実行
cumulative_returns.json を読み込み dsr_results.json / wf_results.json を再生成
"""
import sys, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

try:
    from scipy import stats as sc_stats
    from scipy.special import ndtr
    from scipy.optimize import brentq
except ImportError:
    print("ERROR: pip install scipy"); sys.exit(1)

_CG_ROOT = Path(__file__).parent.parent
OUTPUT   = _CG_ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

# ── 共通指標計算（INVESTMENT_METRICS_REFERENCE.md準拠）────────────
def calc_stats(rets):
    r = pd.Series([float(x) for x in rets]).dropna()
    n = len(r)
    if n < 6: return {}
    cagr    = float((1+r).prod()**(12/n)-1)
    sharpe  = float(r.mean()/r.std(ddof=1)*np.sqrt(12)) if r.std(ddof=1)>0 else 0
    # Fix 2026-03-21: 半分散Sortino — 負リターン月のみ（r[r<0]）で計算
    neg     = r[r < 0]
    ds      = float(np.sqrt((neg**2).mean())*np.sqrt(12)) if len(neg)>0 else 0
    sortino = float(r.mean()*12/ds) if ds>0 else 0
    cum     = (1+r).cumprod()
    maxdd   = float(((cum-cum.cummax())/cum.cummax()).min())
    calmar  = abs(cagr/maxdd) if maxdd!=0 else 0
    return dict(cagr=round(cagr,4), sharpe=round(sharpe,4),
                sortino=round(sortino,4), maxdd=round(maxdd,4), calmar=round(calmar,4))

# ── DSR (Bailey & Lopez de Prado 2014) ───────────────────────────
def dsr_value(sharpe, n_trials, T):
    gamma = 0.5772156649
    emax  = (1-gamma)*sc_stats.norm.ppf(1-1/n_trials) + gamma*sc_stats.norm.ppf(1-1/(n_trials*np.e)) if n_trials>1 else 0
    sr_m  = sharpe / np.sqrt(12)
    sr_bm = emax   / np.sqrt(12)
    var_sr = (1 + (3-1)/4*sr_m**2) / (T-1) if T>1 else 0
    if var_sr <= 0: return 1.0
    z = (sr_m - sr_bm) / np.sqrt(var_sr)
    return float(ndtr(z))

# ── Block Bootstrap (H0: SR=0) ────────────────────────────────────
def bootstrap_pval(rets, n_boot=10000, block_len=12, seed=42):
    r = np.array([float(x) for x in rets])
    T = len(r)
    obs_sr = float(r.mean()/r.std(ddof=1)*np.sqrt(12)) if r.std(ddof=1)>0 else 0
    rng = np.random.default_rng(seed)
    r_dm = r - r.mean()
    count = 0
    n_blocks = int(np.ceil(T / block_len))
    starts = np.arange(T - block_len + 1)
    for _ in range(n_boot):
        idx  = rng.choice(starts, size=n_blocks)
        boot = np.concatenate([r_dm[i:i+block_len] for i in idx])[:T]
        bsr  = float(boot.mean()/boot.std(ddof=1)*np.sqrt(12)) if boot.std(ddof=1)>0 else 0
        if bsr >= obs_sr: count += 1
    return count / n_boot

# ── Walk-Forward (Expanding Window) ──────────────────────────────
def run_wf(rets_all, dates_all, train_min=36, test_months=12):
    rets = [float(x) for x in rets_all]
    N, results = len(rets), []
    i = train_min
    while i + test_months <= N:
        tr = calc_stats(rets[:i])
        te = calc_stats(rets[i:i+test_months])
        if tr and te:
            results.append({
                'test':     f"{dates_all[i]}~{dates_all[i+test_months-1]}",
                'train_n':  i,
                'test_cagr':    te['cagr'],
                'test_sharpe':  te['sharpe'],
                'test_sortino': te['sortino'],
                'test_maxdd':   te['maxdd'],
                'train_sharpe': tr['sharpe'],
            })
        i += test_months
    return results

def main():
    cg = json.load(open(OUTPUT / 'cumulative_returns.json'))
    T  = len(cg['dates'])
    RATIOS = ['r0_100','r10_90','r20_80','r30_70','r40_60','r50_50',
              'r60_40','r70_30','r80_20','r90_10','r100_0']
    N_TRIALS = len(RATIOS)

    # ── DSR再生成 ────────────────────────────────────────────────
    print(f"DSR計算中 (N={N_TRIALS}, T={T})...")
    dsr_results = []
    for key in RATIOS:
        rets = cg.get(f'ret_{key}', [])
        s = calc_stats(rets)
        sh = s.get('sharpe', 0)
        dsr_val = dsr_value(sh, N_TRIALS, T)
        pval    = bootstrap_pval(rets, n_boot=10000, block_len=12)
        e_pct   = int(key.split('_')[0][1:])
        d_pct   = 100 - e_pct
        dsr_results.append(dict(
            key=key, label=f'ETF{e_pct}%+DEF{d_pct}%',
            etf_pct=e_pct, def_pct=d_pct, T=T,
            CAGR=round(s.get('cagr',0),4), Sharpe=round(sh,4),
            MaxDD=round(s.get('maxdd',0),4),
            DSR=round(dsr_val,4), p_val=round(pval,4),
            DSR_pass=bool(dsr_val>=0.95),
            Boot_pass=bool(pval<0.05),
            BOTH_pass=bool(dsr_val>=0.95 and pval<0.05)
        ))
        print(f"  {key}: Sharpe={sh:.4f} DSR={dsr_val:.4f} p={pval:.4f}")

    dsr_out = {
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'N': N_TRIALS, 'T': T,
        'note': 'DSR: Bailey & Lopez de Prado 2014 / Bootstrap: block=12 n=10000 / 日次防御型統一',
        'results': dsr_results
    }
    with open(OUTPUT / 'dsr_results.json', 'w') as f:
        json.dump(dsr_out, f, indent=2)
    print(f"✅ dsr_results.json saved (T={T})")

    # ── WF再生成 ─────────────────────────────────────────────────
    # 実際の窓数を計算して表示
    _sample_wf = run_wf(list(cg.get('ret_r20_80',[])), cg['dates'])
    _n_win = len(_sample_wf)
    print(f"WF計算中 ({_n_win} windows)...")
    wf_out = {
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'method': 'Expanding Window', 'train_min_months': 36,
        'test_months': 12, 'step_months': 12, 'n_windows': _n_win,
        'ratios': {}
    }
    for key in RATIOS:
        rets = cg.get(f'ret_{key}', [])
        wf   = run_wf(rets, cg['dates'])
        is_s = cg['stats'].get(key, {}).get('sharpe', 0)
        oos_s = [w['test_sharpe'] for w in wf]
        oos_c = [w['test_cagr']   for w in wf]
        oos_d = [w['test_maxdd']  for w in wf]
        wf_out['ratios'][key] = {
            'is_sharpe':     round(is_s, 4),
            'oos_sharpe_avg': round(float(np.mean(oos_s)), 4),
            'oos_sharpe_min': round(float(np.min(oos_s)),  4),
            'oos_cagr_avg':   round(float(np.mean(oos_c)), 4),
            'oos_cagr_min':   round(float(np.min(oos_c)),  4),
            'oos_maxdd_avg':  round(float(np.mean(oos_d)), 4),
            'oos_maxdd_worst':round(float(np.min(oos_d)),  4),
            'gap_sharpe':     round(float(np.mean(oos_s)) - is_s, 4),
            'windows': [{'test':w['test'], 'train_n':w['train_n'],
                          'cagr':w['test_cagr'], 'sharpe':w['test_sharpe'],
                          'sortino':w['test_sortino'], 'maxdd':w['test_maxdd']}
                         for w in wf]
        }
    with open(OUTPUT / 'wf_results.json', 'w') as f:
        json.dump(wf_out, f, indent=2)
    print(f"✅ wf_results.json saved")

if __name__ == '__main__':
    main()
