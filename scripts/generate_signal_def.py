#!/usr/bin/env python3
"""
Combined Grail — OG防御型 Signal Generator
ETF14銘柄の6Mモメンタム Top4 + InvVol加重
"""
import sys, json, warnings
from pathlib import Path
from datetime import datetime, date
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

ETFS = ['GLD','EEM','IWM','EFA','QQQ','AGG','TLT','IEF','SHY','DBC','TIP','IYR','LQD']
ETF_NAMES = {
    'GLD':'SPDR Gold Shares','EEM':'iShares MSCI Emerging Markets',
    'IWM':'iShares Russell 2000','EFA':'iShares MSCI EAFE',
    'QQQ':'Invesco QQQ','AGG':'iShares Core US Aggregate Bond',
    'TLT':'iShares 20+ Year Treasury','IEF':'iShares 7-10 Year Treasury',
    'SHY':'iShares 1-3 Year Treasury','DBC':'Invesco DB Commodity',
    'TIP':'iShares TIPS Bond','IYR':'iShares US Real Estate','LQD':'iShares iBoxx Investment Grade',
}
TOP_N = 4

def main():
    today    = date.today()
    sig_date = f"{today.year}-{today.month:02d}"
    print(f"OG防御型 Signal — {sig_date}")

    print(f"  {len(ETFS)}銘柄の価格取得中...")
    raw = yf.download(ETFS, period='9mo', interval='1mo',
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Close' in raw.columns.get_level_values(0):
            prices = raw['Close']
        elif 'Close' in raw.columns.get_level_values(1):
            prices = raw.xs('Close', axis=1, level=1)
        else:
            prices = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        prices = raw[['Close']] if 'Close' in raw.columns else raw
    prices.index = pd.to_datetime(prices.index)

    # 6Mモメンタム
    mom = {}
    for t in ETFS:
        if t not in prices.columns: continue
        ps = prices[t].dropna()
        if len(ps) < 7: continue
        mom[t] = float(ps.iloc[-1]/ps.iloc[-7] - 1)

    top4 = sorted(mom.items(), key=lambda x: -x[1])[:TOP_N]
    selected = [t for t,_ in top4]
    print(f"  Top4: {selected}")

    # InvVol加重（日次90日）
    daily = yf.download(selected, period='6mo', interval='1d',
                        auto_adjust=True, progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        if 'Close' in daily.columns.get_level_values(0):
            dp = daily['Close']
        elif 'Close' in daily.columns.get_level_values(1):
            dp = daily.xs('Close', axis=1, level=1)
        else:
            dp = daily.xs(daily.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        dp = daily[['Close']] if 'Close' in daily.columns else daily

    vols = {}
    for t in selected:
        if t not in dp.columns: continue
        rets = dp[t].dropna().pct_change().dropna().iloc[-63:]
        if len(rets) >= 10:
            v = float(rets.std() * (252**0.5))
            if v > 0: vols[t] = v

    if vols:
        ti = sum(1/v for v in vols.values())
        weights = {t: (1/v)/ti for t,v in vols.items()}
    else:
        weights = {t: 1/len(selected) for t in selected}

    # 現在価格
    prices_now = {}
    for t in selected:
        try: prices_now[t] = round(float(yf.Ticker(t).fast_info.last_price), 2)
        except: prices_now[t] = 0.0

    signal = {
        'date': sig_date,
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy': 'OG防御型',
        'top_n': TOP_N,
        'holdings': [
            {'ticker': t, 'name': ETF_NAMES.get(t,t),
             'momentum_6m': round(mom.get(t,0),4),
             'weight': round(weights.get(t,0),4),
             'price': prices_now.get(t,0)}
            for t in selected
        ],
    }

    path = OUTPUT / 'signal_def_latest.json'
    with open(path,'w') as f: json.dump(signal,f,indent=2,ensure_ascii=False)
    print(f"✅ {path}")
    return signal

if __name__ == '__main__':
    main()
