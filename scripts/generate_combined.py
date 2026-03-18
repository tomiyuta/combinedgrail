#!/usr/bin/env python3
"""
Combined Grail — 統合シグナル生成
ETF M4 × OG防御型 の配分比率で統合
"""
import sys, json, subprocess, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
warnings.filterwarnings('ignore')

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'
SCRIPTS= ROOT / 'scripts'
OUTPUT.mkdir(exist_ok=True)

# デフォルト配分比率
ETF_WEIGHT = 0.40   # HolyETF M4
DEF_WEIGHT = 0.60   # OG防御型 (ETF Top4)

def run_sub(script):
    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: {script.name} failed:\n{result.stderr[:300]}")

def main():
    print(f"\n{'='*50}")
    print(f"Combined Grail Signal Generator")
    print(f"{'='*50}")

    # 各シグナルを生成
    print("\n[1/2] HolyETF M4 シグナル生成...")
    run_sub(SCRIPTS / 'generate_signal_etf.py')

    print("[2/2] OG防御型 シグナル生成...")
    run_sub(SCRIPTS / 'generate_signal_def.py')

    # 読み込み
    etf_sig = json.load(open(OUTPUT / 'signal_etf_latest.json'))
    def_sig = json.load(open(OUTPUT / 'signal_def_latest.json'))

    date_str = etf_sig['date']

    # 統合銘柄リスト（比率調整後）
    combined_holdings = []
    for h in etf_sig['holdings']:
        combined_holdings.append({
            'ticker':   h['ticker'],
            'strategy': 'ETF M4',
            'weight_in_strategy': h['weight'],
            'weight_in_portfolio': round(h['weight'] * ETF_WEIGHT, 4),
            'price':    h.get('price', 0),
        })
    for h in def_sig['holdings']:
        combined_holdings.append({
            'ticker':   h['ticker'],
            'name':     h.get('name', h['ticker']),
            'strategy': 'OG防御型',
            'momentum_6m': h.get('momentum_6m', 0),
            'weight_in_strategy': h['weight'],
            'weight_in_portfolio': round(h['weight'] * DEF_WEIGHT, 4),
            'price':    h.get('price', 0),
        })

    # 価格更新（for rebalance calc）
    combined_holdings.sort(key=lambda x: -x['weight_in_portfolio'])

    signal = {
        'date':        date_str,
        'generated':   datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'etf_weight':  ETF_WEIGHT,
        'def_weight':  DEF_WEIGHT,
        'etf_signal':  etf_sig,
        'def_signal':  def_sig,
        'holdings':    combined_holdings,
        'regime':      etf_sig.get('regime', '—'),
        'regime_label': {
            'CAUTIOUS':             '強気（攻撃型）',
            'PREEMPTIVE':           '警戒（守備準備）',
            'STRESS':               'ストレス（守備型）',
            'ACUTE':                '急性（フル守備）',
            'ACUTE_EQUITY':         '急性株式（回復型）',
            'RATES_UP_HV_RECOVERY': '金利上昇・株回復',
            'RATES_UP_HV_INFLATION':'金利上昇・インフレ',
            'RATES_UP_LOW_VOL':     '金利上昇・低ボラ',
        }.get(etf_sig.get('regime',''), etf_sig.get('regime','')),
    }

    # 保存
    path = OUTPUT / 'signal_latest.json'
    with open(path, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {path}")
    print(f"   date={date_str} regime={signal['regime']}")
    print(f"   ETF M4 ({ETF_WEIGHT*100:.0f}%): {[h['ticker'] for h in etf_sig['holdings']]}")
    print(f"   OG防御型 ({DEF_WEIGHT*100:.0f}%): {[h['ticker'] for h in def_sig['holdings']]}")
    return signal

if __name__ == '__main__':
    main()
