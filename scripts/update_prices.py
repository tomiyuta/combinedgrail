#!/usr/bin/env python3
"""
Combined Grail — Daily Price Updater
毎日 JST07:00 に株価・為替・SPYレジームを更新
"""
import sys, json, warnings, urllib.request
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: pip install yfinance pandas"); sys.exit(1)

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

TICKERS = [
    'TQQQ','TECL','SOXL','GLD','GLDM','XLU','TMV','TLT','XLV',
    'EEM','IWM','EFA','QQQ','AGG','IEF','SHY','DBC','TIP','IYR','LQD',
    'SPY','QQQ','AAPL','MSFT','NVDA',
]

def fetch_prices(tickers):
    prices = {}
    for t in list(set(tickers)):
        try:
            p = yf.Ticker(t).fast_info.last_price
            if p and p > 0: prices[t] = round(float(p), 2)
        except: pass
    return prices

def fetch_usdjpy():
    try:
        req = urllib.request.Request('https://open.er-api.com/v6/latest/USD',
                                     headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return round(json.load(r)['rates']['JPY'], 2)
    except: return 155.0

def main():
    print(f"Combined Grail Daily Update — {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    prices = fetch_prices(TICKERS)
    usdjpy = fetch_usdjpy()
    print(f"  prices={len(prices)}, USDJPY={usdjpy}")

    out = {
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'usdjpy': usdjpy,
        'prices': prices,
    }
    path = OUTPUT / 'prices_latest.json'
    with open(path,'w') as f: json.dump(out,f,indent=2)
    print(f"✅ {path}")

if __name__ == '__main__':
    main()
