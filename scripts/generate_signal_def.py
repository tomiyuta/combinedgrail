#!/usr/bin/env python3
"""
Combined Grail — OG防御型 Signal Generator
OpenGrail (opengrail.vercel.app) の generate_signal.py と完全同一ロジック

同一仕様:
  - ETFユニバース: DEFENSIVE_ETFS 14銘柄（SPY含む）
  - データ取得: 日次 interval='1d', period='7mo'
  - モメンタム: 日次 126取引日前 (≈6ヶ月)
  - ボラティリティ: 日次リターン 直近90日 年率換算
  - 加重: InvVol (1/vol正規化)
  - Top N: 4銘柄
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

# ── ETFユニバース（OpenGrailと完全同一・14銘柄） ──────────
DEFENSIVE_ETFS = [
    {"symbol":"GLD",  "name":"SPDR Gold Shares",                   "category":"金"},
    {"symbol":"EEM",  "name":"iShares MSCI Emerging Markets",       "category":"新興国株"},
    {"symbol":"IWM",  "name":"iShares Russell 2000",                "category":"小型株"},
    {"symbol":"EFA",  "name":"iShares MSCI EAFE",                   "category":"先進国株"},
    {"symbol":"QQQ",  "name":"Invesco QQQ Trust",                   "category":"NASDAQ100"},
    {"symbol":"SPY",  "name":"SPDR S&P 500 ETF",                   "category":"S&P500"},
    {"symbol":"DBC",  "name":"Invesco DB Commodity",                "category":"コモディティ"},
    {"symbol":"IEF",  "name":"iShares 7-10 Year Treasury",         "category":"中期国債"},
    {"symbol":"LQD",  "name":"iShares Investment Grade Corp Bond",  "category":"投資適格社債"},
    {"symbol":"AGG",  "name":"iShares Core US Aggregate Bond",      "category":"総合債券"},
    {"symbol":"TLT",  "name":"iShares 20+ Year Treasury",          "category":"長期国債"},
    {"symbol":"TIP",  "name":"iShares TIPS Bond",                   "category":"物価連動債"},
    {"symbol":"SHY",  "name":"iShares 1-3 Year Treasury",          "category":"短期国債"},
    {"symbol":"IYR",  "name":"iShares US Real Estate",              "category":"不動産"},
]

TOP_N = 4  # OpenGrailと同一

# ── ユーティリティ関数（OpenGrailと完全同一） ──────────────

def get_multi_daily(tickers: list, period: str = '7mo') -> dict:
    """バッチ取得でAPIコールを削減 — OpenGrail generate_signal.py と同一"""
    try:
        raw = yf.download(tickers, period=period, interval='1d',
                          auto_adjust=True, progress=False, group_by='ticker')
        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    s = raw[t]['Close'].dropna()
                    if len(s) >= 20:
                        result[t] = s.to_frame('close')
                except:
                    pass
        else:
            if len(tickers) == 1 and 'Close' in raw.columns:
                s = raw['Close'].dropna()
                if len(s) >= 20:
                    result[tickers[0]] = s.to_frame('close')
        return result
    except Exception as e:
        print(f"  batch error: {e}")
        return {}

def calc_momentum_6m(df: pd.DataFrame) -> float:
    """日次126取引日前 (≈6ヶ月) モメンタム — OpenGrail と同一"""
    if len(df) < 100: return float('-inf')
    p_now = float(df['close'].iloc[-1])
    p_6m  = float(df['close'].iloc[-126]) if len(df) >= 126 else float(df['close'].iloc[0])
    if p_6m == 0: return float('-inf')
    return (p_now - p_6m) / p_6m

def calc_annvol(df: pd.DataFrame) -> float:
    """日次リターン直近90日・年率換算 — OpenGrail と同一"""
    if len(df) < 20: return 0.0
    rets = df['close'].pct_change().dropna().iloc[-90:]
    if len(rets) < 5: return 0.0
    return float(rets.std() * np.sqrt(252))

def invvol_weights(items: list) -> list:
    """InvVol加重 — OpenGrail と同一"""
    valid = [x for x in items if x.get('risk', 0) > 0]
    if not valid: return items
    total_inv = sum(1/x['risk'] for x in valid)
    for x in valid:
        x['weight'] = round((1/x['risk']) / total_inv, 4)
    return valid

def select_portfolio(data_map: dict, name_map: dict, top_n: int,
                     category_map: dict = None) -> list:
    """モメンタム上位Top-N選択 + InvVol加重 — OpenGrail と同一"""
    metrics = []
    for ticker, df in data_map.items():
        mom = calc_momentum_6m(df)
        vol = calc_annvol(df)
        if mom == float('-inf') or vol <= 0: continue
        item = {
            'ticker':   ticker,
            'name':     name_map.get(ticker, ticker),
            'momentum': round(mom, 4),
            'risk':     round(vol, 4),
        }
        if category_map:
            item['category'] = category_map.get(ticker, '')
        metrics.append(item)

    metrics.sort(key=lambda x: x['momentum'], reverse=True)
    selected = metrics[:top_n]
    selected = invvol_weights(selected)
    for x in selected:
        x.pop('risk', None)
    return selected


def main():
    today    = date.today()
    sig_date = f"{today.year}-{today.month:02d}"
    print(f"OG防御型 Signal (OpenGrail同一ロジック) — {sig_date}")

    # ── データ取得（OpenGrailと同一: 日次 period='7mo'） ──
    def_symbols  = [e['symbol']   for e in DEFENSIVE_ETFS]
    def_name_map = {e['symbol']: e['name']     for e in DEFENSIVE_ETFS}
    def_cat_map  = {e['symbol']: e['category'] for e in DEFENSIVE_ETFS}

    print(f"  {len(def_symbols)}銘柄の日次価格取得中 (period=7mo)...")
    def_data = get_multi_daily(def_symbols, period='7mo')
    print(f"  fetched {len(def_data)} ETFs")

    # ── Top4選択 + InvVol加重（OpenGrailと同一） ──────────
    defensive = select_portfolio(def_data, def_name_map, top_n=TOP_N,
                                 category_map=def_cat_map)
    print(f"  selected: {[x['ticker'] for x in defensive]}")

    signal = {
        'date':      sig_date,
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy':  'OG防御型',
        'top_n':     TOP_N,
        'holdings':  defensive,
    }

    path = OUTPUT / 'signal_def_latest.json'
    with open(path, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)
    print(f"✅ {path}")
    return signal


if __name__ == '__main__':
    main()
