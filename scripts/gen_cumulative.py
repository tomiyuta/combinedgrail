#!/usr/bin/env python3
"""
Combined Grail — Standalone Cumulative Returns Generator
外部リポジトリへの依存なし / GitHub Actions で完全自律実行可能

生成データ:
  - ret_M4    : CombinedGrail/engine/ + CombinedGrail/data/ から自己完結計算
  - ret_OGdef : OG防御型の月次リターン（yfinance から直接取得 / 14銘柄）
  - ret_SPY   : SPYの月次リターン
  - 各ブレンド比率 (0:100 〜 100:0) の累積リターン・統計・年次リターン
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

_CG_ROOT   = Path(__file__).parent.parent
_ENGINE    = _CG_ROOT / 'engine'
OUTPUT     = _CG_ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(_ENGINE))
from signal_generator import generate_signal
from data_loader import DataLoader, DATA_PATHS

# ── OG防御型ユニバース（OpenGrailと同一14銘柄） ──────────────────
OG_ETFS = ['GLD','EEM','IWM','EFA','QQQ','SPY','DBC','IEF',
           'LQD','AGG','TLT','TIP','SHY','IYR']
OG_TOP_N = 4

# ── M4バックテスト対象ETF ────────────────────────────────────────
M4_ETFS  = ['GLD', 'XLU', 'TMV', 'TECL', 'TQQQ', 'SOXL', 'XLV']


# ════════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════════
def stats(rets):
    """半分散Sortino — 負リターン月のみで半分散計算（INVESTMENT_METRICS_REFERENCE.md準拠）
    Fix 2026-03-21: np.where方式（全月N割り）→ r[r<0]方式（負月のみ）に修正
    """
    r = pd.Series([float(x) for x in rets]).dropna()
    if len(r) < 6: return {}
    cagr    = float((1+r).prod()**(12/len(r))-1)
    sh      = float(r.mean()/r.std(ddof=1)*12**0.5) if r.std()>0 else 0
    neg     = r[r < 0]
    ds      = float(np.sqrt((neg**2).mean()) * 12**0.5) if len(neg) > 0 else 0
    sortino = float(r.mean()*12/ds) if ds>0 else 0
    cum     = (1+r).cumprod()
    md      = float(((cum-cum.cummax())/cum.cummax()).min())
    ca      = abs(cagr/md) if md != 0 else 0
    return dict(cagr=round(cagr,4), sharpe=round(sh,4),
                sortino=round(sortino,4), maxdd=round(md,4),
                calmar=round(ca,4))

def annual_rets(dates, rets):
    df = pd.DataFrame({'date': pd.to_datetime(dates, format='%Y-%m'), 'r': rets})
    df['year'] = df['date'].dt.year
    result = {}
    for y, g in df.groupby('year'):
        result[str(y)] = round(float((1+pd.Series(g['r'].values)).prod()-1), 4)
    return result

def cum_from_rets(rets):
    c = [1.0]
    for r in rets: c.append(round(c[-1]*(1+float(r)), 6))
    return c[1:]


# ════════════════════════════════════════════════════════════════
# Step 1: M4 月次リターン計算（CombinedGrail/engine + data）
# ════════════════════════════════════════════════════════════════
def compute_m4_returns(start='2016-01'):
    logging.info("[1/3] M4月次リターン計算...")
    today     = date.today()
    end_label = f"{today.year}-{today.month:02d}"

    dl = DataLoader(); dl.load()

    # 特徴量構築
    g = pd.DataFrame(index=pd.period_range(start, end_label, freq='M'))
    for tk, w in [('LQD',6),('LQD',3),('BIL',6),('BIL',3),('TLT',3),('TLT',6),
                  ('SPY',6),('SPY',3),('TQQQ',3),('TQQQ',12),
                  ('XLU',6),('GLD',6),('TECL',12)]:
        try: g[f'{tk}_{w}m'] = dl.get_rolling_return(tk, w)
        except: pass
    g['VIX']       = dl.vix
    g['LQD_SHY_6m'] = g.get('LQD_6m', pd.Series(0, index=g.index)) \
                     - g.get('BIL_6m', pd.Series(0, index=g.index))
    g['LQD_SHY_3m'] = g.get('LQD_3m', pd.Series(0, index=g.index)) \
                     - g.get('BIL_3m', pd.Series(0, index=g.index))
    g['XLU_SPY_3m'] = g.get('XLU_6m', pd.Series(0, index=g.index)) \
                     - g.get('SPY_6m',  pd.Series(0, index=g.index))

    # C5指標
    try:
        ki = pd.read_csv(DATA_PATHS['phase_c'], index_col=0, parse_dates=True)
        ki.index = ki.index.to_period('M')
        ki['SKEW_norm']  = (ki['SKEW'] - ki['SKEW'].mean()) / ki['SKEW'].std()
        ki['slope_norm'] = (ki['term_slope'] - ki['term_slope'].mean()) / ki['term_slope'].std()
        g = g.join(ki[['SKEW_norm','slope_norm']], how='left')
    except Exception as e:
        logging.warning(f"  C5データskip: {e}")
    g = g.dropna(subset=['LQD_SHY_6m','VIX','TLT_6m','SPY_6m']).sort_index()

    # 月次価格→リターン
    prices_raw = pd.read_csv(DATA_PATHS['prices'], index_col=0, parse_dates=True)
    prices_raw.index = prices_raw.index.to_period('M')
    ret_m = prices_raw.pct_change()

    # 各月: シグナル → ウェイト → リターン
    dates_out, ret_m1_list, ret_m3_list, ret_m4_list, ret_spy_list = [], [], [], [], []
    regimes_out = []

    for period in g.index:
        row  = g.loc[period]
        sig  = generate_signal(row.to_dict())
        # Fix 2026-03-21: look-ahead bias修正 — 当月末シグナル→翌月リターン
        next_period = period + 1
        if next_period not in ret_m.index:
            continue
        r_now = ret_m.loc[next_period]

        def portfolio_ret(weights):
            return float(sum(weights.get(etf, 0) * r_now.get(etf, 0)
                             for etf in weights))

        dates_out.append(str(period))   # シグナル生成月をラベルに使用
        ret_m1_list.append(round(portfolio_ret(sig['weights_M1']), 6))
        ret_m3_list.append(round(portfolio_ret(sig['weights_M3']), 6))
        ret_m4_list.append(round(portfolio_ret(sig['weights_M4']), 6))
        ret_spy_list.append(round(float(r_now.get('SPY', 0)), 6))
        regimes_out.append(sig['regime'])

    logging.info(f"  M4: {dates_out[0]} ~ {dates_out[-1]} ({len(dates_out)}ヶ月)")
    return dates_out, ret_m1_list, ret_m3_list, ret_m4_list, ret_spy_list, regimes_out


# ════════════════════════════════════════════════════════════════
# Step 2: OG防御型 月次リターン計算（yfinanceから直接取得）
# ════════════════════════════════════════════════════════════════
def compute_ogdef_returns(dates_m4):
    """
    OG防御型バックテスト — generate_signal_def.py / opengrail generate_signal.py と完全同一方式
    - データ: 日次 interval='1d'（月次ではなく日次で6Mモメンタム計算）
    - ユニバース: OG_ETFS 14銘柄（SPY含む）
    - 選択: Top4 + InvVol加重（日次リターン90日stdで計算）
    - generate_signal.py の select_portfolio と同一ロジック
    """
    logging.info("[2/3] OG防御型 月次リターン計算（日次データ・generate_signal同一方式）...")
    start_dt = dates_m4[0]
    end_dt   = dates_m4[-1]

    # 日次価格取得（2015-01-01から: 6Mモメンタム計算に必要なバッファ込み）
    raw = yf.download(OG_ETFS, start='2015-01-01',
                      end=f"{end_dt[:4]}-{min(int(end_dt[5:])+2, 12):02d}-01",
                      interval='1d', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices_d = raw['Close'] if 'Close' in raw.columns.get_level_values(0)                    else raw.xs('Close', axis=1, level=1)
    else:
        prices_d = raw

    prices_d.index = pd.to_datetime(prices_d.index)

    def_rets = {}
    for period_str in dates_m4:
        yr, mo = int(period_str[:4]), int(period_str[5:7])
        # 月末の日次データを取得
        month_end = prices_d[prices_d.index.year==yr][prices_d[prices_d.index.year==yr].index.month==mo]
        if month_end.empty:
            def_rets[period_str] = 0.0
            continue
        last_idx = prices_d.index.get_loc(month_end.index[-1])

        # ── 6Mモメンタム（日次126取引日前）── generate_signal.py と同一
        if last_idx < 126:
            def_rets[period_str] = 0.0
            continue
        p_now = prices_d.iloc[last_idx]
        p_6m  = prices_d.iloc[last_idx - 126]
        mom = {}
        for t in OG_ETFS:
            if t in p_now.index and t in p_6m.index:
                pn, p6 = float(p_now[t]), float(p_6m[t])
                if p6 > 0 and not np.isnan(pn) and not np.isnan(p6):
                    mom[t] = (pn - p6) / p6

        top_n_sel = sorted(mom.items(), key=lambda x: -x[1])[:OG_TOP_N]
        selected  = [t for t, _ in top_n_sel]
        if not selected:
            def_rets[period_str] = 0.0
            continue

        # ── InvVol加重（日次リターン直近90日・年率換算）── generate_signal.py と同一
        vols = {}
        for t in selected:
            if t in prices_d.columns:
                r_hist = prices_d[t].iloc[max(0, last_idx-90):last_idx].pct_change().dropna()
                if len(r_hist) >= 10:
                    v = float(r_hist.std() * np.sqrt(252))
                    if v > 0: vols[t] = v
        if vols:
            ti = sum(1/v for v in vols.values())
            weights = {t: (1/v)/ti for t, v in vols.items()}
        else:
            weights = {t: 1/len(selected) for t in selected}

        # ── 当月リターン（月末→翌月末の価格変化）──
        # 翌月末を取得
        next_yr, next_mo = (yr, mo+1) if mo < 12 else (yr+1, 1)
        next_month = prices_d[(prices_d.index.year==next_yr) & (prices_d.index.month==next_mo)]
        if next_month.empty:
            def_rets[period_str] = 0.0
            continue
        next_idx = prices_d.index.get_loc(next_month.index[-1])
        p_end = prices_d.iloc[next_idx]

        month_ret = 0.0
        for t in selected:
            if t in p_now.index and t in p_end.index:
                p0, p1 = float(p_now[t]), float(p_end[t])
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    month_ret += weights.get(t, 0) * (p1 - p0) / p0
        def_rets[period_str] = round(month_ret, 6)

    result = [def_rets.get(d, 0.0) for d in dates_m4]
    logging.info(f"  OG防御型: {dates_m4[0]} ~ {dates_m4[-1]} ({len(result)}ヶ月)")
    return result


# ════════════════════════════════════════════════════════════════
# Step 3: ブレンド・統計・出力
# ════════════════════════════════════════════════════════════════
def main():
    logging.info(f"gen_cumulative.py — {datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}")

    dates, ret_m1, ret_m3, ret_m4, ret_spy, regimes = compute_m4_returns()
    ret_def = compute_ogdef_returns(dates)

    logging.info("[3/3] ブレンド・統計計算...")
    ratios = [(0,100),(10,90),(20,80),(30,70),(40,60),(50,50),
              (60,40),(70,30),(80,20),(90,10),(100,0)]

    out = {
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'dates':     dates,
        'regimes':   regimes,
        'M1':  cum_from_rets(ret_m1),
        'M3':  cum_from_rets(ret_m3),
        'M4':  cum_from_rets(ret_m4),
        'SPY': cum_from_rets(ret_spy),
        'defense': cum_from_rets(ret_def),
        'ret_M1':  [round(r,6) for r in ret_m1],
        'ret_M3':  [round(r,6) for r in ret_m3],
        'ret_M4':  [round(r,6) for r in ret_m4],
        'ret_SPY': [round(r,6) for r in ret_spy],
        'ret_defense': [round(r,6) for r in ret_def],
    }

    # 各ブレンド比率
    ret_all = {}
    for e_pct, d_pct in ratios:
        e, d = e_pct/100, d_pct/100
        rets = [e*re + d*rd for re, rd in zip(ret_m4, ret_def)]
        key  = f'r{e_pct}_{d_pct}'
        out[key]          = cum_from_rets(rets)
        out[f'ret_{key}'] = [round(r,6) for r in rets]
        ret_all[key]      = rets
    out['ret_SPY_blend'] = [round(r,6) for r in ret_spy]

    # 統計（M1/M3/M4 + ブレンド全11 + 防御型単体 + SPY）
    out['stats'] = {}
    for key, rets in [('M1',ret_m1),('M3',ret_m3),('M4',ret_m4),
                      ('defense',ret_def),('SPY',ret_spy)]:
        out['stats'][key] = stats(rets)
    for key, rets in ret_all.items():
        out['stats'][key] = stats(rets)

    # 年次リターン
    out['annual'] = {}
    for key, rets in [('M1',ret_m1),('M3',ret_m3),('M4',ret_m4),
                      ('defense',ret_def),('SPY',ret_spy)]:
        out['annual'][key] = annual_rets(dates, rets)
    for key, rets in ret_all.items():
        out['annual'][key] = annual_rets(dates, rets)

    path = OUTPUT / 'cumulative_returns.json'
    with open(path, 'w') as f:
        json.dump(out, f)

    s4 = out['stats'].get('r40_60', {})
    s0 = out['stats'].get('r100_0', {})
    sd = out['stats'].get('defense', {})
    logging.info(f"  ETF単体(r100_0): CAGR={s0.get('cagr',0):.1%} Sharpe={s0.get('sharpe',0):.2f} MaxDD={s0.get('maxdd',0):.1%}")
    logging.info(f"  推奨(r40_60):    CAGR={s4.get('cagr',0):.1%} Sharpe={s4.get('sharpe',0):.2f} MaxDD={s4.get('maxdd',0):.1%}")
    logging.info(f"  防御型(r0_100):  CAGR={sd.get('cagr',0):.1%} Sharpe={sd.get('sharpe',0):.2f} MaxDD={sd.get('maxdd',0):.1%}")
    logging.info(f"✅ {path}")
    return out

if __name__ == '__main__':
    main()
