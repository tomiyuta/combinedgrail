#!/usr/bin/env python3
"""
Combined Grail — HolyETF M4 Signal Generator (本家エンジン直接使用)
HolyGrail_ETF/engine/signal_generator.py + data_loader.py を直接インポート
差異ゼロ: 同一ロジック・同一データ・同一出力
"""
import sys, os, json, warnings, logging
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ── CombinedGrail/engine を sys.path に追加（GitHub Actions対応）──
_CG_ROOT    = Path(__file__).parent.parent
_ETF_ENGINE = _CG_ROOT / 'engine'
sys.path.insert(0, str(_ETF_ENGINE))

try:
    from signal_generator import (
        regime_gate, credit_sensor, rates_sensor,
        rotation_sensor, momentum_sensor, mode_router,
        compute_weights, generate_signal, c5_check, rates_acute,
        DISCRETE
    )
    from data_loader import DataLoader, DATA_PATHS
    logging.info("✅ HolyETF本家エンジン読み込み成功")
except ImportError as e:
    logging.error(f"❌ HolyETFエンジン読み込み失敗: {e}")
    sys.exit(1)

OUTPUT = _CG_ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

REGIME_LABELS = {
    'CAUTIOUS':             '強気（攻撃型）',
    'PREEMPTIVE':           '慎重（転換前）',
    'STRESS':               '警戒（防御型）',
    'ACUTE':                '急性（防御型）',
    'ACUTE_EQUITY':         '急性株式（回復型）',
    'RATES_UP_HV_RECOVERY': '金利上昇・HV回復',
    'RATES_UP_HV_INFLATION':'金利上昇・インフレ',
    'RATES_UP_LOW_VOL':     '金利上昇・低ボラ',
}

def main():
    today    = date.today()
    sig_date = f"{today.year}-{today.month:02d}"
    logging.info(f"HolyETF M4 Signal (本家エンジン) — {sig_date}")

    # ── DataLoader でデータ読み込み（HolyETF本家と同一） ──────────
    dl = DataLoader()
    dl.load()

    # ── 特徴量構築（本家と同一） ────────────────────────────────────
    g = pd.DataFrame(index=pd.period_range('2016-01', sig_date, freq='M'))
    for tk, w in [('LQD',6),('LQD',3),('BIL',6),('BIL',3),('TLT',3),('TLT',6),
                  ('SPY',6),('SPY',3),('TQQQ',3),('TQQQ',12),('XLU',6),('GLD',6),('TECL',12)]:
        try:
            g[f'{tk}_{w}m'] = dl.get_rolling_return(tk, w)
        except Exception as e:
            logging.warning(f"  {tk}_{w}m: skip ({e})")

    g['VIX'] = dl.vix
    g['LQD_SHY_6m'] = g.get('LQD_6m', pd.Series(0,index=g.index)) - \
                      g.get('BIL_6m', pd.Series(0,index=g.index))
    g['LQD_SHY_3m'] = g.get('LQD_3m', pd.Series(0,index=g.index)) - \
                      g.get('BIL_3m', pd.Series(0,index=g.index))
    g['XLU_SPY_3m'] = g.get('XLU_6m', pd.Series(0,index=g.index)) - \
                      g.get('SPY_6m',  pd.Series(0,index=g.index))

    # ── C5 vol指標 ─────────────────────────────────────────────────
    try:
        ki = pd.read_csv(DATA_PATHS['phase_c'], index_col=0, parse_dates=True)
        ki.index = ki.index.to_period('M')
        ki['SKEW_norm']  = (ki['SKEW']       - ki['SKEW'].mean())       / ki['SKEW'].std()
        ki['slope_norm'] = (ki['term_slope'] - ki['term_slope'].mean()) / ki['term_slope'].std()
        g = g.join(ki[['SKEW_norm','slope_norm']], how='left')
    except Exception as e:
        logging.warning(f"  phase_c skip: {e}")

    g = g.dropna(subset=['LQD_SHY_6m','VIX','TLT_6m','SPY_6m']).sort_index()
    latest = g.iloc[-1]

    # ── 本家generate_signal呼び出し ────────────────────────────────
    sig = generate_signal(latest.to_dict())
    sig['date'] = sig_date

    logging.info(f"  Regime: {sig['regime']}")
    s = sig['sensors']
    logging.info(f"  Sensors: CREDIT={s['CREDIT']} RATES={s['RATES']} "
                 f"ROTATION={s['ROTATION']} MOMENTUM={s['MOMENTUM']}")
    logging.info(f"  Mode: {sig['mode']}  C5={sig['c5_active']}  "
                 f"rates_acute={sig['rates_acute']}")

    # ── M4配分 ────────────────────────────────────────────────────
    w4 = sig['weights_M4']
    holdings = [
        {'ticker': k, 'weight': round(v, 4),
         'strategy': 'ETF M4',
         'weight_in_strategy': round(v, 4)}
        for k, v in w4.items() if v > 0.001
    ]
    holdings.sort(key=lambda x: -x['weight'])

    output = {
        'date':         sig_date,
        'generated':    datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'regime':       sig['regime'],
        'regime_label': REGIME_LABELS.get(sig['regime'], sig['regime']),
        'sensors':      sig['sensors'],
        'mode':         sig['mode'],
        'c5_active':    sig['c5_active'],
        'rates_acute':  sig['rates_acute'],
        'weights_M1':   sig['weights_M1'],
        'weights_M3':   sig['weights_M3'],
        'weights_M4':   sig['weights_M4'],
        'holdings':     holdings,
    }

    for path in [OUTPUT / f"signal_etf_{sig_date.replace('-','_')}.json",
                 OUTPUT / 'signal_etf_latest.json']:
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"  ✅ Saved signal_etf_latest.json  regime={output['regime']}")
    return output

if __name__ == '__main__':
    main()
