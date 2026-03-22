#!/usr/bin/env python3
"""CombinedGrail — signal_latest.json 統合生成スクリプト
α（BAM劇薬DM）+ OG防御型 を既存スキーマ互換形式で結合"""
import json, sys, logging
from pathlib import Path
from datetime import datetime
from generate_signal_alpha import main as gen_alpha

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'

GATE_LABELS = {
    'ATK': {'en':'Attack',  'ja':'1st gate通過'},
    'DEF': {'en':'Defence', 'ja':'1st gate閉鎖'},
}
LAYER_DESC = {
    'A': 'K-gate（LQD信用ゲート）',
    'B': '朱雀レイヤー（TMV/Tech）',
    'C': 'XLUレイヤー（守備継続）',
    'D': 'Tech選択（TECL/TQQQ）',
}

def main():
    logging.info('generate_signal_combined.py — 統合シグナル生成')

    # ── α シグナル生成 ─────────────────────────────────────
    alpha = gen_alpha()
    gate  = alpha['gate']   # 'ATK' or 'DEF'
    sz    = alpha['sz']
    layers= alpha['layers'] # {'A':gate,'B':...,'C':...,'D':...}
    alloc = alpha['alloc']  # e.g. {'XLU':50,'TQQQ':50}
    feat  = alpha['features']
    alert = alpha['alert']
    bnd_l = alpha['boundary_LQD']
    bnd_t = alpha['boundary_tech']

    # α → 既存 etf_signal スキーマに変換（センサー名をA/B/C/D化）
    etf_signal = {
        'date':        alpha['date'],
        'generated':   alpha['generated'],
        'regime':      gate,
        'regime_label': GATE_LABELS[gate]['ja'],
        'sensors': {
            'A': layers['A'],   # K-gate
            'B': layers['B'],   # 朱雀
            'C': layers['C'],   # XLU
            'D': layers['D'],   # Tech
        },
        'mode':        f'sz={sz:.2f}',
        'c5_active':   alert,
        'rates_acute': bnd_l,
        'holdings': alpha['holdings'],
        # M4互換 (α単体をM4としても格納)
        'weights_M4': {tk: w/100 for tk, w in alloc.items()},
        'features':    feat,
        'boundary_LQD':  bnd_l,
        'boundary_tech': bnd_t,
        'sz':            sz,
    }

    # ── OG防御型シグナル読み込み ────────────────────────────
    def_path = OUTPUT / 'signal_def_latest.json'
    def_signal = {}
    if def_path.exists():
        def_signal = json.loads(def_path.read_text())
        logging.info(f'  OG防御型シグナル読み込み: {def_path}')

    # ── ポートフォリオ合算（α40% + OG60%） ─────────────────
    ETF_W = 0.40; DEF_W = 0.60
    combined_holdings = []
    holding_map = {}
    for h in alpha['holdings']:
        tk = h['ticker']
        holding_map[tk] = holding_map.get(tk,0) + h['weight']*ETF_W
    for h in def_signal.get('holdings', []):
        tk = h['ticker']
        holding_map[tk] = holding_map.get(tk,0) + h['weight']*DEF_W
    for tk, w in sorted(holding_map.items(), key=lambda x:-x[1]):
        combined_holdings.append({'ticker':tk, 'weight':round(w,4)})

    # ── 統合JSON ───────────────────────────────────────────
    out = {
        'date':        alpha['date'],
        'generated':   datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'etf_weight':  ETF_W,
        'def_weight':  DEF_W,
        'regime':      gate,
        'regime_label': GATE_LABELS[gate]['ja'],
        'etf_signal':  etf_signal,
        'def_signal':  def_signal,
        'holdings':    combined_holdings,
        # α-Grail 固有フィールド
        'alpha_gate':  gate,
        'alpha_sz':    sz,
        'alpha_layers': layers,
        'alpha_alloc': alloc,
        'alert':       alert,
    }

    path = OUTPUT / 'signal_latest.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logging.info(f'  ✅ {path}')
    logging.info(f'  Gate={gate}  sz={sz}  ETF:{ETF_W*100:.0f}% / DEF:{DEF_W*100:.0f}%')
    logging.info(f'  combined holdings: {[h["ticker"] for h in combined_holdings]}')
    return out

if __name__ == '__main__': main()
