#!/usr/bin/env python3
"""CombinedGrail — 月次リターン追記スクリプト (Web完結版 / 案B)
既存 cumulative_returns.json に新しい月分を追記する。
ローカル依存ゼロ: yfinance + FRED API のみ使用。

【設計 (案B)】
  - 過去の累積データは既存JSONをそのまま使用（正確値を維持）
  - 毎月1日のCI実行で「前月のリターン」を1ヶ月分だけ追記
  - α月次リターン = 前月末シグナルに基づく配分 × 当月実績リターン
  - OG防御型リターン = 6Mモメンタム Top4 × 当月実績リターン
"""
import json, warnings, logging, io, urllib.request
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

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
CUM_JSON = OUTPUT / 'cumulative_returns.json'

FRED_BASE  = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='
OG_ETFS    = ['GLD','EEM','IWM','EFA','QQQ','SPY','DBC','IEF',
              'LQD','AGG','TLT','TIP','SHY','IYR']
OG_TOP_N   = 4
ALPHA_ETFS = ['LQD','TECL','TQQQ','XLU','GLD','TMV','BIL']

def fetch_fred(series_id):
    url = FRED_BASE + series_id
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                     na_values='.').iloc[:,0].dropna()
    return df.resample('MS').last()

def fetch_prices_daily(tickers, years=2):
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    raw = yf.download(tickers, start=start, interval='1d',
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw['Close'] if 'Close' in raw.columns.get_level_values(0) \
               else raw.xs('Close', axis=1, level=1)
    return raw

def compute_alpha_signal(signal_month, prices_daily, dtb3):
    """前月末時点のαシグナルを計算（配分比率を返す）"""
    # signal_month = pd.Timestamp of signal generation month (e.g. 2026-02-01)
    # 前月末の日次データで月次リターン計算
    sig_yr, sig_mo = signal_month.year, signal_month.month
    prev_mo_data = prices_daily[
        (prices_daily.index.year == sig_yr) &
        (prices_daily.index.month == sig_mo)
    ]
    if prev_mo_data.empty:
        return None

    last_idx = len(prices_daily.loc[:prev_mo_data.index[-1]])

    def ret_lb(tk, lb_days):
        if tk not in prices_daily.columns: return float('nan')
        s = prices_daily[tk].dropna()
        pos = s.index.get_loc(prev_mo_data.index[-1]) if prev_mo_data.index[-1] in s.index else None
        if pos is None or pos < lb_days: return float('nan')
        p_now, p_lb = float(s.iloc[pos]), float(s.iloc[pos - lb_days])
        return (p_now - p_lb) / p_lb if p_lb != 0 else float('nan')

    lqd_12m = ret_lb('LQD', 252)
    prev_ts = signal_month - pd.DateOffset(months=1)
    dtb_v   = float(dtb3.loc[prev_ts]) if prev_ts in dtb3.index else float('nan')
    lqd_ex  = lqd_12m - dtb_v if not (np.isnan(lqd_12m) or np.isnan(dtb_v)) else float('nan')

    gate = 'ATK' if (not np.isnan(lqd_ex) and lqd_ex > 0) else 'DEF'

    if gate == 'DEF':
        return {'XLU': 1.0}

    rel_tq  = ret_lb('TECL', 252) - ret_lb('TQQQ', 252)
    mom4m   = ret_lb('TMV', 84)
    suzaku  = (not np.isnan(mom4m) and mom4m > -0.05)
    tech    = ('TECL' if (not np.isnan(rel_tq) and rel_tq > 0.07
                          and not np.isnan(lqd_ex) and lqd_ex > 0.01) else 'TQQQ')
    if suzaku: return {'XLU': 0.5, tech: 0.5}
    return {'GLD': 1/3, 'XLU': 1/3, tech: 1/3}

def compute_og_signal(signal_month, prices_daily):
    """前月末時点のOG防御型シグナルを計算（Top4 InvVol加重）"""
    sig_yr, sig_mo = signal_month.year, signal_month.month
    mo_data = prices_daily[
        (prices_daily.index.year == sig_yr) &
        (prices_daily.index.month == sig_mo)
    ]
    if mo_data.empty: return None
    last_day = mo_data.index[-1]
    pos_last = prices_daily.index.get_loc(last_day)
    if pos_last < 126: return None

    p_now = prices_daily.iloc[pos_last]
    p_6m  = prices_daily.iloc[pos_last - 126]
    mom = {t: (float(p_now[t]) - float(p_6m[t])) / float(p_6m[t])
           for t in OG_ETFS
           if t in p_now.index and float(p_6m.get(t,0)) > 0
           and not np.isnan(p_now.get(t, float('nan')))}

    top_sel = [t for t,_ in sorted(mom.items(), key=lambda x:-x[1])[:OG_TOP_N]]
    if not top_sel: return None

    vols = {}
    for t in top_sel:
        rc = prices_daily[t].dropna().pct_change().dropna()
        recent = rc.iloc[max(0, pos_last-90):pos_last]
        vols[t] = float(recent.std()) if len(recent) > 10 else 0.01
    inv_vol = {t: 1.0/max(v,1e-6) for t,v in vols.items()}
    total   = sum(inv_vol.values())
    return {t: inv_vol[t]/total for t in top_sel}

def compute_month_return(alloc, return_month, prices_daily):
    """当月の配分×実績リターンを計算"""
    ret_yr, ret_mo = return_month.year, return_month.month
    mo_data = prices_daily[
        (prices_daily.index.year == ret_yr) &
        (prices_daily.index.month == ret_mo)
    ]
    if len(mo_data) < 2: return 0.0
    p_start = prices_daily.iloc[prices_daily.index.get_loc(mo_data.index[0]) - 1] \
              if prices_daily.index.get_loc(mo_data.index[0]) > 0 else mo_data.iloc[0]
    p_end   = mo_data.iloc[-1]

    total = 0.0
    for tk, w in alloc.items():
        if tk in p_start.index and tk in p_end.index:
            p0, p1 = float(p_start[tk]), float(p_end[tk])
            if p0 > 0 and not np.isnan(p1):
                total += w * (p1 - p0) / p0
    return round(float(total), 6)

def cum_from_rets(rets):
    c = [1.0]
    for r in rets: c.append(round(c[-1]*(1+float(r)), 6))
    return c[1:]

def stats(rets):
    r = pd.Series([float(x) for x in rets]).dropna()
    if len(r) < 6: return {}
    n = len(r); cagr=(1+r).prod()**(12/n)-1
    sh = r.mean()/r.std(ddof=1)*12**0.5 if r.std()>0 else 0
    neg = r[r<0]; ds = np.sqrt((neg**2).mean())*12**0.5 if len(neg)>0 else 0
    sortino = r.mean()*12/ds if ds>0 else 0
    cum = (1+r).cumprod(); md = ((cum-cum.cummax())/cum.cummax()).min()
    ca = abs(cagr/md) if md!=0 else 0
    return dict(cagr=round(float(cagr),4), sharpe=round(float(sh),4),
                sortino=round(float(sortino),4), maxdd=round(float(md),4),
                calmar=round(float(ca),4))

def annual_rets(dates, rets):
    df = pd.DataFrame({'d':pd.to_datetime(dates,format='%Y-%m'),'r':rets})
    df['y'] = df['d'].dt.year
    return {str(y): round(float((1+pd.Series(g['r'].values)).prod()-1),4)
            for y,g in df.groupby('y')}

def main():
    logging.info('append_monthly_return.py (web) — 月次リターン追記')

    # ── 既存JSONを読み込み ─────────────────────────────────────
    with open(CUM_JSON) as f: data = json.load(f)
    last_date  = pd.Timestamp(data['dates'][-1] + '-01')
    today      = pd.Timestamp.now().replace(day=1)
    prev_month = today - pd.DateOffset(months=1)  # 前月 = 追記対象月

    if last_date >= prev_month:
        logging.info(f'  追記不要: last={last_date.strftime("%Y-%m")} '
                     f'prev={prev_month.strftime("%Y-%m")}')
        return

    logging.info(f'  追記対象: {prev_month.strftime("%Y-%m")}')

    # ── 全銘柄の日次価格取得 ──────────────────────────────────
    all_tickers = list(set(ALPHA_ETFS + OG_ETFS))
    logging.info(f'  yfinance {len(all_tickers)}銘柄 取得中...')
    prices_d = fetch_prices_daily(all_tickers, years=2)
    dtb3     = fetch_fred('DTB3') / 100

    # ── 前月のシグナル計算（シグナル月 = 追記対象月）─────────────
    # 例: 2026-03月のリターン → シグナルは2026-03末に生成 → 翌月2026-04に反映
    # 毎月1日のCI: prev_month = 2026-02 → シグナル月=2026-02 → 実績月=2026-02
    signal_month = prev_month  # シグナル生成月（追記月と同じ）

    alpha_alloc = compute_alpha_signal(signal_month, prices_d, dtb3)
    og_alloc    = compute_og_signal(signal_month, prices_d)

    if alpha_alloc is None or og_alloc is None:
        logging.warning('  シグナル計算失敗 → スキップ')
        return

    logging.info(f'  α alloc: {alpha_alloc}')
    logging.info(f'  OG alloc: {og_alloc}')

    # ── 実績リターン計算（追記対象月）────────────────────────────
    ret_a   = compute_month_return(alpha_alloc, prev_month, prices_d)
    ret_def = compute_month_return(og_alloc,    prev_month, prices_d)
    ret_spy = compute_month_return({'SPY':1.0},  prev_month, prices_d)
    new_label = prev_month.strftime('%Y-%m')

    logging.info(f'  {new_label}: α={ret_a:.4f}  OG={ret_def:.4f}  SPY={ret_spy:.4f}')

    # ── JSONに追記 ─────────────────────────────────────────────
    data['dates'].append(new_label)
    data['ret_alpha'].append(ret_a)
    data['ret_defense'].append(ret_def)
    data['ret_SPY'].append(ret_spy)

    ratios = [(0,100),(10,90),(20,80),(30,70),(40,60),(50,50),
              (60,40),(70,30),(80,20),(90,10),(100,0)]
    for e_pct, d_pct in ratios:
        e, d = e_pct/100, d_pct/100
        blended = round(ret_a*e + ret_def*d, 6)
        data[f'ret_r{e_pct}_{d_pct}'].append(blended)

    # 累積・統計・年次を全再計算
    data['alpha']   = cum_from_rets(data['ret_alpha'])
    data['defense'] = cum_from_rets(data['ret_defense'])
    data['SPY']     = cum_from_rets(data['ret_SPY'])
    for e_pct, d_pct in ratios:
        key = f'r{e_pct}_{d_pct}'
        data[key] = cum_from_rets(data[f'ret_{key}'])

    data['stats'] = {}
    data['annual'] = {}
    for key, rets in [('alpha',data['ret_alpha']),('defense',data['ret_defense']),('SPY',data['ret_SPY'])]:
        data['stats'][key]  = stats(rets)
        data['annual'][key] = annual_rets(data['dates'], rets)
    for e_pct, d_pct in ratios:
        key = f'r{e_pct}_{d_pct}'
        data['stats'][key]  = stats(data[f'ret_{key}'])
        data['annual'][key] = annual_rets(data['dates'], data[f'ret_{key}'])

    data['generated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(CUM_JSON, 'w') as f: json.dump(data, f)
    logging.info(f'  ✅ {CUM_JSON} 更新完了 (T={len(data["dates"])}M)')

if __name__ == '__main__': main()
