#!/usr/bin/env python3
"""
Combined Grail — HolyETF M4 Signal Generator
HolyGrail_ETFのB5v2配分テーブルをベースにyfinanceで4センサー判定
"""
import sys, json, warnings, logging
from pathlib import Path
from datetime import datetime, date
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

# ══ B5v2 配分テーブル ══
DISCRETE = {
    'CAUTIOUS':             {'TQQQ':.35,'TECL':.20,'SOXL':.15,'XLU':.20,'GLD':.10},
    'PREEMPTIVE':           {'XLU':.25,'XLV':.15,'GLD':.15,'TQQQ':.25,'TECL':.10,'TMV':.10},
    'STRESS':               {'GLD':.30,'TMV':.30,'XLU':.30,'TQQQ':.05,'TECL':.05},
    'ACUTE':                {'GLD':.30,'TMV':.40,'XLU':.30},
    'ACUTE_EQUITY':         {'TQQQ':.35,'TECL':.25,'GLD':.25,'XLU':.15},
    'RATES_UP_HV_RECOVERY': {'TQQQ':.40,'TECL':.30,'GLD':.20,'XLU':.10},
    'RATES_UP_HV_INFLATION':{'GLD':.40,'XLU':.25,'TLT':.20,'XLV':.15},
    'RATES_UP_LOW_VOL':     {'TMV':.40,'GLD':.20,'XLU':.20,'TQQQ':.10,'TECL':.10},
}

def fetch(ticker, period='6mo', interval='1d'):
    d = yf.download(ticker, period=period, interval=interval,
                    auto_adjust=True, progress=False)
    return d['Close'].squeeze().dropna()

def calc_sensors():
    """4センサー（CREDIT/RATES/ROTATION/MOMENTUM）を yfinance で計算"""
    logging.info("センサー計算中...")

    # CREDIT: LQD(社債)/IEF(国債) 3M相対リターン
    lqd = fetch('LQD','6mo'); ief = fetch('IEF','6mo')
    lqd_ret = float(lqd.iloc[-1]/lqd.iloc[-63]-1) if len(lqd)>=63 else 0
    ief_ret = float(ief.iloc[-1]/ief.iloc[-63]-1) if len(ief)>=63 else 0
    credit = 'ATK' if lqd_ret >= ief_ret else 'DEF'

    # RATES: 長短金利差 (TLT/SHY 3M相対リターン)
    tlt = fetch('TLT','6mo'); shy = fetch('SHY','6mo')
    tlt_ret = float(tlt.iloc[-1]/tlt.iloc[-63]-1) if len(tlt)>=63 else 0
    shy_ret = float(shy.iloc[-1]/shy.iloc[-63]-1) if len(shy)>=63 else 0
    # TLT上昇=金利低下=RATES ATK(通常)、TLT下落=金利上昇=RATES DEF
    rates_raw = 'ATK' if tlt_ret >= shy_ret else 'DEF'
    # 金利急騰判定（TLT 1ヶ月 -3%以上下落）
    tlt_1m = float(tlt.iloc[-1]/tlt.iloc[-22]-1) if len(tlt)>=22 else 0
    rates_acute = tlt_1m < -0.03

    # ROTATION: XLU(ディフェンシブ)/SPY 1M相対強度
    xlu = fetch('XLU','3mo'); spy = fetch('SPY','3mo')
    xlu_1m = float(xlu.iloc[-1]/xlu.iloc[-22]-1) if len(xlu)>=22 else 0
    spy_1m = float(spy.iloc[-1]/spy.iloc[-22]-1) if len(spy)>=22 else 0
    rotation = 'DEF' if xlu_1m > spy_1m else 'ATK'

    # MOMENTUM: SPY 6M
    spy6 = fetch('SPY','9mo')
    spy_6m = float(spy6.iloc[-1]/spy6.iloc[-126]-1) if len(spy6)>=126 else 0
    momentum = 'ATK' if spy_6m > 0 else 'DEF'

    logging.info(f"  CREDIT={credit} RATES={rates_raw}(acute={rates_acute}) "
                 f"ROTATION={rotation} MOMENTUM={momentum}")
    return dict(CREDIT=credit, RATES=rates_raw, ROTATION=rotation,
                MOMENTUM=momentum, rates_acute=rates_acute)

def determine_regime(sensors):
    c = sensors['CREDIT']; ra = sensors['RATES']
    rot = sensors['ROTATION']; mom = sensors['MOMENTUM']
    acute = sensors['rates_acute']

    if c == 'DEF' and ra == 'DEF':
        return 'STRESS'
    if c == 'DEF' and ra == 'ATK':
        return 'ACUTE' if mom == 'DEF' else 'ACUTE_EQUITY'
    # credit ATK
    if acute:
        return 'RATES_UP_HV_INFLATION' if rot == 'DEF' else 'RATES_UP_HV_RECOVERY'
    if ra == 'DEF':
        return 'RATES_UP_LOW_VOL'
    # CREDIT ATK, RATES ATK, no acute
    return 'CAUTIOUS' if mom == 'ATK' else 'PREEMPTIVE'

def main():
    today    = date.today()
    sig_date = f"{today.year}-{today.month:02d}"
    logging.info(f"HolyETF M4 Signal — {sig_date}")

    sensors = calc_sensors()
    regime  = determine_regime(sensors)
    weights = DISCRETE.get(regime, DISCRETE['CAUTIOUS'])
    logging.info(f"  regime={regime} → {weights}")

    # 現在価格
    tickers = list(weights.keys())
    prices  = {}
    for t in tickers:
        try:
            prices[t] = round(float(yf.Ticker(t).fast_info.last_price), 2)
        except: prices[t] = 0.0

    signal = {
        'date': sig_date,
        'generated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy': 'HolyETF M4',
        'regime': regime,
        'sensors': {k:v for k,v in sensors.items() if k != 'rates_acute'},
        'rates_acute': sensors['rates_acute'],
        'holdings': [
            {'ticker': t, 'weight': round(w,4), 'price': prices.get(t,0)}
            for t,w in sorted(weights.items(), key=lambda x:-x[1])
        ],
    }

    path = OUTPUT / 'signal_etf_latest.json'
    with open(path,'w') as f: json.dump(signal,f,indent=2,ensure_ascii=False)
    logging.info(f"✅ {path}")
    return signal

if __name__ == '__main__':
    main()
