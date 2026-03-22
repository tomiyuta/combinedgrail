#!/usr/bin/env python3
"""CombinedGrail — αシグナル生成 (Web完結版)
yfinance + FRED API で全データ取得 → ローカル依存ゼロ
"""
import sys, json, warnings, logging, io
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
import urllib.request
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

# ── FRED 無料CSV API（APIキー不要）──────────────────────────────
FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='

ETF_DESC = {
    'TECL':'テクノロジー 3倍レバレッジ', 'TQQQ':'NASDAQ 3倍レバレッジ',
    'XLU':'公益セクター（守備）',         'GLD':'金・クライシスヘッジ',
    'TMV':'長期国債 逆3倍（金利上昇ヘッジ）',
}
ETF_COLORS = {
    'TECL':'#F59E0B','TQQQ':'#60A5FA','XLU':'#34D399',
    'GLD':'#D4AF7A','TMV':'#A78BFA',
}
PRICE_TICKERS = ['LQD','TECL','TQQQ','XLU','GLD','TMV']

def fetch_fred(series_id):
    """FRED CSVを取得 → 月次DatetimeIndexのSeries"""
    url = FRED_BASE + series_id
    logging.info(f'  FRED {series_id} 取得中...')
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                     na_values='.').iloc[:,0].dropna()
    return df.resample('MS').last()

def fetch_prices(tickers, years=2):
    """yfinanceで月次終値取得（直近2年分で十分）"""
    from datetime import timedelta
    start = (date.today().replace(day=1) - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    logging.info(f'  yfinance {tickers} 取得中... (start={start})')
    raw = yf.download(tickers, start=start, interval='1mo',
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'] if 'Close' in raw.columns.get_level_values(0) \
                 else raw.xs('Close', axis=1, level=1)
    else:
        prices = raw
    prices.index = prices.index.to_period('M').to_timestamp()
    return prices

def load_data():
    prices = fetch_prices(PRICE_TICKERS)
    dtb3   = fetch_fred('DTB3') / 100      # 年率 → 小数
    hy     = fetch_fred('BAMLH0A0HYM2')
    return prices, dtb3, hy

def compute_features(ts, prices, dtb3, hy):
    prev = ts - pd.DateOffset(months=1)
    def ret(tk, lb):
        if tk not in prices.columns: return float('nan')
        s = prices[tk].dropna()
        if prev not in s.index: return float('nan')
        idx = s.index.get_loc(prev)
        if idx < lb: return float('nan')
        p_now, p_lb = float(s.iloc[idx]), float(s.iloc[idx-lb])
        return (p_now - p_lb) / p_lb if p_lb != 0 else float('nan')
    lqd_12m = ret('LQD', 12)
    dtb_v   = float(dtb3.loc[prev]) if prev in dtb3.index else float('nan')
    lqd_ex  = lqd_12m - dtb_v if not (np.isnan(lqd_12m) or np.isnan(dtb_v)) else float('nan')
    hy_v    = float(hy.loc[prev]) if prev in hy.index else None
    return {
        'LQD_ex':        round(float(lqd_ex), 4) if not np.isnan(lqd_ex) else None,
        'LQD_12m':       round(float(lqd_12m), 4) if not np.isnan(lqd_12m) else None,
        'rel_TQ_12m':    round(ret('TECL',12) - ret('TQQQ',12), 4),
        'mom4m_TMV':     round(ret('TMV', 4), 4),
        'XLU_12m':       round(ret('XLU', 12), 4),
        'HY_spread_pct': round(float(hy_v), 2) if hy_v is not None else None,
    }

def compute_signals(f):
    lqd_ex = f['LQD_ex']; rel_tq = f['rel_TQ_12m']
    mom4m  = f['mom4m_TMV']; hy_v  = f['HY_spread_pct']
    gate = 'ATK' if (lqd_ex is not None and lqd_ex > 0) else 'DEF'
    if gate == 'DEF':                               sz = 0.0
    elif lqd_ex is not None and abs(lqd_ex) < 0.01: sz = 0.5
    else:                                            sz = 1.0
    if hy_v is not None and hy_v > 7.0: sz *= 0.7
    suzaku = (mom4m is not None and mom4m > -0.05)
    tech   = ('TECL' if (rel_tq is not None and rel_tq > 0.07
                         and lqd_ex is not None and lqd_ex > 0.01) else 'TQQQ')
    has_xlu= not (f.get('XLU_12m') is not None and f.get('XLU_12m',1) <= 0.01
                  and lqd_ex is not None and lqd_ex > 0.03)
    bnd_l  = (lqd_ex is not None and abs(lqd_ex) < 0.01)
    bnd_t  = (rel_tq is not None and abs(rel_tq) < 0.05)
    if gate == 'DEF':       alloc = {'XLU': 100.0}
    elif suzaku:            alloc = {'XLU': 50.0, tech: 50.0}
    else:                   alloc = {'GLD': 33.3, 'XLU': 33.3, tech: 33.3}
    holdings = [
        {'ticker': tk, 'name': ETF_DESC.get(tk, tk),
         'weight': round(w/100, 4), 'color': ETF_COLORS.get(tk,'#888')}
        for tk, w in alloc.items()
    ]
    return dict(gate=gate, sz=round(sz,2),
                tech_choice=tech, has_xlu=has_xlu,
                boundary_LQD=bnd_l, boundary_tech=bnd_t,
                alert=bnd_l or bnd_t, alloc=alloc, holdings=holdings,
                layers={'A':gate, 'B':'TMV' if suzaku else 'Tech',
                        'C':'XLU' if has_xlu else 'OFF', 'D':tech})

def main():
    today  = date.today()
    target = pd.Timestamp(f'{today.year}-{today.month:02d}-01')
    label  = target.strftime('%Y-%m')
    logging.info(f'generate_signal_alpha.py (web) — {label}')
    prices, dtb3, hy = load_data()
    f   = compute_features(target, prices, dtb3, hy)
    sig = compute_signals(f)
    out = {
        'date': label, 'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy': 'α', 'gate': sig['gate'], 'sz': sig['sz'],
        'layers': sig['layers'], 'alloc': sig['alloc'], 'holdings': sig['holdings'],
        'alert': sig['alert'], 'boundary_LQD': sig['boundary_LQD'],
        'boundary_tech': sig['boundary_tech'], 'features': f,
    }
    for path in [OUTPUT/'signal_etf_latest.json',
                 OUTPUT/f'signal_etf_{label.replace("-","_")}.json']:
        with open(path,'w') as fp: json.dump(out, fp, indent=2, ensure_ascii=False)
    logging.info(f'  Gate={sig["gate"]}  sz={sig["sz"]}  alloc={sig["alloc"]}')
    return out

if __name__ == '__main__': main()
