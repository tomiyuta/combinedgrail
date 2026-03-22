#!/usr/bin/env python3
"""
CombinedGrail — Cumulative Returns Generator (α版)
M4エンジン → BAM α（劇薬DM）に置換
OG防御型・ブレンド・SPYはそのまま継続
"""
import sys, os, json, warnings, logging
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

_CG_ROOT = Path(__file__).parent.parent
OUTPUT   = _CG_ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)
BAM      = Path('/Users/yutatomi/Downloads/01_投資・定量分析/BAM')

OG_ETFS  = ['GLD','EEM','IWM','EFA','QQQ','SPY','DBC','IEF',
            'LQD','AGG','TLT','TIP','SHY','IYR']
OG_TOP_N = 4

def stats(rets):
    r = pd.Series([float(x) for x in rets]).dropna()
    if len(r) < 6: return {}
    cagr    = float((1+r).prod()**(12/len(r))-1)
    sh      = float(r.mean()/r.std(ddof=1)*12**0.5) if r.std()>0 else 0
    neg     = r[r<0]
    ds      = float(np.sqrt((neg**2).mean())*12**0.5) if len(neg)>0 else 0
    sortino = float(r.mean()*12/ds) if ds>0 else 0
    cum     = (1+r).cumprod()
    md      = float(((cum-cum.cummax())/cum.cummax()).min())
    ca      = abs(cagr/md) if md!=0 else 0
    return dict(cagr=round(cagr,4), sharpe=round(sh,4), sortino=round(sortino,4),
                maxdd=round(md,4), calmar=round(ca,4))

def annual_rets(dates, rets):
    df = pd.DataFrame({'date':pd.to_datetime(dates,format='%Y-%m'),'r':rets})
    df['year'] = df['date'].dt.year
    result = {}
    for y, g in df.groupby('year'):
        result[str(y)] = round(float((1+pd.Series(g['r'].values)).prod()-1),4)
    return result

def cum_from_rets(rets):
    c = [1.0]
    for r in rets: c.append(round(c[-1]*(1+float(r)),6))
    return c[1:]


# ── Step 1: α（BAM劇薬DM）月次リターン計算 ──────────────────────
def compute_alpha_returns():
    logging.info("[1/3] α（BAM劇薬DM）月次リターン計算...")
    prices   = pd.read_csv(BAM/'06_市場データ/prices_monthly_2004_2026.csv',
                            index_col=0, parse_dates=True)
    dtb3     = pd.read_csv(BAM/'06_市場データ/FRED/FRED_DTB3_20260306.csv',
                            index_col=0, parse_dates=True).iloc[:,0].resample('MS').last()/100
    holdings = pd.read_csv(BAM/'06_市場データ/market/strategy_holdings_monthly.csv',
                            parse_dates=['month'])
    orig     = holdings[holdings['strategy']=='GEKIYAKU_ORIG'].pivot_table(
                index='month', columns='asset', values='observed_weight', fill_value=0)
    lqd_ex   = (prices['LQD'].pct_change(12) - dtb3.reindex(prices.index, method='ffill')).dropna()

    ret_alpha, ret_spy_list, dates_out = [], [], []
    for ts in orig.index:
        if ts not in prices.index: continue
        r = sum(orig.loc[ts,a]*prices[a].pct_change(1).loc[ts]
                for a in orig.columns if a in prices.columns
                and not pd.isna(prices[a].pct_change(1).loc[ts]))
        lqd_v = lqd_ex.reindex([ts], method='ffill').iloc[0] if ts in lqd_ex.index else 0
        ret_alpha.append(float(r))
        ret_spy_list.append(float(prices['SPY'].pct_change(1).loc[ts])
                             if ts in prices.index else 0.0)
        dates_out.append(ts.strftime('%Y-%m'))

    logging.info(f"  α: {dates_out[0]} ~ {dates_out[-1]} ({len(dates_out)}ヶ月)")
    return dates_out, ret_alpha, ret_spy_list

# ── Step 2: OG防御型（既存ロジックそのまま流用）──────────────────
def compute_ogdef_returns(dates_alpha):
    logging.info("[2/3] OG防御型 月次リターン計算...")
    start_dt = dates_alpha[0]; end_dt = dates_alpha[-1]
    raw = yf.download(OG_ETFS, start='2009-01-01',
                      end=f"{end_dt[:4]}-{min(int(end_dt[5:])+2,12):02d}-01",
                      interval='1d', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices_d = raw['Close'] if 'Close' in raw.columns.get_level_values(0) \
                   else raw.xs('Close', axis=1, level=1)
    else:
        prices_d = raw
    prices_d.index = pd.to_datetime(prices_d.index)

    def_rets = {}
    for period_str in dates_alpha:
        yr, mo = int(period_str[:4]), int(period_str[5:7])
        month_end = prices_d[(prices_d.index.year==yr) & (prices_d.index.month==mo)]
        if month_end.empty: def_rets[period_str]=0.0; continue
        last_idx = prices_d.index.get_loc(month_end.index[-1])
        if last_idx < 126: def_rets[period_str]=0.0; continue
        p_now = prices_d.iloc[last_idx]; p_6m = prices_d.iloc[last_idx-126]
        mom = {}
        for t in OG_ETFS:
            if t in p_now.index and t in p_6m.index:
                pn,p6 = float(p_now[t]), float(p_6m[t])
                if p6>0 and not np.isnan(pn) and not np.isnan(p6):
                    mom[t] = (pn-p6)/p6
        top_sel = [t for t,_ in sorted(mom.items(), key=lambda x:-x[1])[:OG_TOP_N]]
        if not top_sel: def_rets[period_str]=0.0; continue
        # InvVol加重
        vols = {}
        for t in top_sel:
            rc = prices_d[t].pct_change().dropna()
            recent = rc.iloc[max(0,last_idx-90):last_idx]
            vols[t] = float(recent.std()) if len(recent)>10 else 0.01
        inv_vol = {t: 1.0/max(v,1e-6) for t,v in vols.items()}
        total_iv = sum(inv_vol.values())
        weights  = {t: inv_vol[t]/total_iv for t in top_sel} if total_iv>0 else {}
        # 翌月リターン
        next_yr = yr if mo<12 else yr+1; next_mo = mo+1 if mo<12 else 1
        next_month = prices_d[(prices_d.index.year==next_yr) & (prices_d.index.month==next_mo)]
        if next_month.empty: def_rets[period_str]=0.0; continue
        next_idx = prices_d.index.get_loc(next_month.index[-1])
        p_end = prices_d.iloc[next_idx]
        month_ret = 0.0
        for t in top_sel:
            if t in p_now.index and t in p_end.index:
                p0,p1 = float(p_now[t]), float(p_end[t])
                if p0>0 and not np.isnan(p0) and not np.isnan(p1):
                    month_ret += weights.get(t,0)*(p1-p0)/p0
        def_rets[period_str] = round(month_ret, 6)

    result = [def_rets.get(d, 0.0) for d in dates_alpha]
    logging.info(f"  OG防御型: {dates_alpha[0]} ~ {dates_alpha[-1]} ({len(result)}ヶ月)")
    return result


# ── Step 3: ブレンド・統計・出力 ──────────────────────────────────
def main():
    logging.info(f"gen_cumulative_alpha.py — {datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}")
    dates, ret_alpha, ret_spy = compute_alpha_returns()
    ret_def = compute_ogdef_returns(dates)
    logging.info("[3/3] ブレンド・統計計算...")

    ratios = [(0,100),(10,90),(20,80),(30,70),(40,60),(50,50),
              (60,40),(70,30),(80,20),(90,10),(100,0)]
    out = {
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'dates':   dates,
        'alpha':   cum_from_rets(ret_alpha),
        'defense': cum_from_rets(ret_def),
        'SPY':     cum_from_rets(ret_spy),
        'ret_alpha':   [round(r,6) for r in ret_alpha],
        'ret_defense': [round(r,6) for r in ret_def],
        'ret_SPY':     [round(r,6) for r in ret_spy],
    }
    # ブレンド全11比率
    ret_all = {}
    for e_pct, d_pct in ratios:
        e, d = e_pct/100, d_pct/100
        rets = [e*ra + d*rd for ra,rd in zip(ret_alpha, ret_def)]
        key  = f'r{e_pct}_{d_pct}'
        out[key]          = cum_from_rets(rets)
        out[f'ret_{key}'] = [round(r,6) for r in rets]
        ret_all[key]      = rets
    out['ret_SPY_blend'] = [round(r,6) for r in ret_spy]

    out['stats']  = {}
    out['annual'] = {}
    for key, rets in [('alpha',ret_alpha),('defense',ret_def),('SPY',ret_spy)]:
        out['stats'][key]  = stats(rets)
        out['annual'][key] = annual_rets(dates, rets)
    for key, rets in ret_all.items():
        out['stats'][key]  = stats(rets)
        out['annual'][key] = annual_rets(dates, rets)

    path = OUTPUT / 'cumulative_returns.json'
    with open(path, 'w') as f: json.dump(out, f)
    s4 = out['stats'].get('r40_60', {})
    logging.info(f"  α40:OG60 CAGR={s4.get('cagr',0):.1%} Sharpe={s4.get('sharpe',0):.3f} MDD={s4.get('maxdd',0):.1%}")
    logging.info(f"✅ {path}")
    return out

if __name__ == '__main__': main()
