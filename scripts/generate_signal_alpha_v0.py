#!/usr/bin/env python3
"""
CombinedGrail — αシグナル生成 (3-Sub EqualWeight FoF版)
設計書: ALPHA_FoF_DESIGN_v1.1.md
バージョン: v2.0 (Phase1+2)
旧版: generate_signal_alpha_v0.py

変更点:
  旧: 単層エンジン (LQD_ex binary gate → 直接配分)
  新: 3-Sub EqualWeight FoF (Sub-A/B/C独立判断 → 1/3票合算)

不変制約:
  [INV-1] yakuzai_v1 は parallel monitor 専用。Subの入口ゲートに使わない。
  [INV-2] 3-Subは独立。共通変数を渡してもロジックは各Sub固有。
  [INV-3] FoF合算は厳密1/3等重。動的ウェイト禁止。
  [INV-4] Phase1+2は不可分。部分評価禁止。

戻り値スキーマ: generate_signal_combined.py との互換を維持
"""
import sys, json, warnings, logging, io
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
from collections import Counter
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

ETF_DESC = {
    'TECL': 'テクノロジー 3倍レバレッジ', 'TQQQ': 'NASDAQ 3倍レバレッジ',
    'XLU':  '公益セクター（守備）',        'GLD':  '金・クライシスヘッジ',
    'TMV':  '長期国債 逆3倍（金利上昇ヘッジ）',
}
ETF_COLORS = {
    'TECL': '#F59E0B', 'TQQQ': '#60A5FA', 'XLU': '#34D399',
    'GLD':  '#D4AF7A', 'TMV':  '#A78BFA',
}
PRICE_TICKERS = ['LQD', 'TECL', 'TQQQ', 'XLU', 'GLD', 'TMV', 'SPY']
FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='


# ══════════════════════════════════════════════════════════════════
# Sub共通インターフェース
# ══════════════════════════════════════════════════════════════════
@dataclass
class SubOutput:
    """
    各Subの独立判断出力。
    tech: 0=守備票 / 1=攻撃票
    choice: TECL/TQQQ (tech=1時。Noneは多数決に委ねる)
    retreat: XLU/GLD (tech=0時)
    """
    tech:    int
    choice:  Optional[str] = None
    retreat: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# データ取得
# ══════════════════════════════════════════════════════════════════
def fetch_fred(series_id: str) -> pd.Series:
    url = FRED_BASE + series_id
    logging.info(f'  FRED {series_id} 取得中...')
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                     na_values='.').iloc[:, 0].dropna()
    return df.resample('MS').last()


def fetch_prices(tickers: list, years: int = 2) -> pd.DataFrame:
    start = (date.today().replace(day=1) - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    logging.info(f'  yfinance {tickers} 取得中...')
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
    dtb3   = fetch_fred('DTB3') / 100
    hy     = fetch_fred('BAMLH0A0HYM2')
    vix    = fetch_fred('VXVCLS')   # [FIX-4追加]
    return prices, dtb3, hy, vix


# ══════════════════════════════════════════════════════════════════
# 特徴量計算
# ══════════════════════════════════════════════════════════════════
def compute_features(ts: pd.Timestamp, prices: pd.DataFrame,
                     dtb3: pd.Series, hy: pd.Series, vix: pd.Series) -> dict:
    """t-1規約: 前月末シグナル → 当月適用"""
    prev = ts - pd.DateOffset(months=1)

    def ret(tk: str, lb: int) -> Optional[float]:
        if tk not in prices.columns: return None
        s = prices[tk].dropna()
        if prev not in s.index: return None
        idx = s.index.get_loc(prev)
        if idx < lb: return None
        pn = float(s.iloc[idx]); pl = float(s.iloc[idx - lb])
        return (pn - pl) / pl if pl != 0 else None

    # 既存特徴量
    lqd_12m = ret('LQD', 12)
    dtb_v   = float(dtb3.loc[prev]) if prev in dtb3.index else None
    # [FIX-1] is not None で統一（0.0や負値で偽判定されないよう修正）
    lqd_ex  = (lqd_12m - dtb_v) if (lqd_12m is not None and dtb_v is not None) else None
    hy_v    = float(hy.loc[prev]) if prev in hy.index else None
    tecl_12m = ret('TECL', 12); tqqq_12m = ret('TQQQ', 12)
    rel_tq_12m = ((tecl_12m - tqqq_12m)
                  if (tecl_12m is not None and tqqq_12m is not None) else None)

    # 新規追加特徴量（Phase1+2必須）
    vix_v   = float(vix.loc[prev]) if prev in vix.index else None
    spy_12m = ret('SPY', 12)
    spy_3m  = ret('SPY', 3)    # yakuzai_v1用 [FIX-4]
    lqd_6m  = ret('LQD', 6)
    tqqq_3m = ret('TQQQ', 3)
    xlu_3m  = ret('XLU', 3);  xlu_6m = ret('XLU', 6)
    gld_3m  = ret('GLD', 3);  gld_6m = ret('GLD', 6)
    tecl_6m = ret('TECL', 6); tecl_3m = ret('TECL', 3)  # [FIX-4]
    tqqq_6m = ret('TQQQ', 6)
    rel_tq_6m = ((tecl_6m - tqqq_6m)
                 if (tecl_6m is not None and tqqq_6m is not None) else None)

    return {
        'LQD_ex': lqd_ex, 'LQD_12m': lqd_12m, 'rel_TQ_12m': rel_tq_12m,
        'mom4m_TMV': ret('TMV', 4), 'XLU_12m': ret('XLU', 12), 'HY_spread_pct': hy_v,
        'VIX': vix_v, 'SPY_12m': spy_12m, 'SPY_3m': spy_3m,
        'LQD_6m': lqd_6m, 'TQQQ_3m': tqqq_3m,
        'XLU_3m': xlu_3m, 'XLU_6m': xlu_6m,
        'GLD_3m': gld_3m, 'GLD_6m': gld_6m,
        'TECL_6m': tecl_6m, 'TECL_3m': tecl_3m,
        'TQQQ_6m': tqqq_6m, 'rel_TQ_6m': rel_tq_6m,
        '_ts': ts,  # Sub-A四半期判定用
    }


# ══════════════════════════════════════════════════════════════════
# Sub-A proxy: §494 四半期逆張りロジック（精度93.2%）
# ══════════════════════════════════════════════════════════════════
def subA_proxy(features: dict, ts: pd.Timestamp) -> SubOutput:
    """
    [FIX-2] Sub-Aは ATK/DEF のみ判定。tech銘柄選択は持たない（choice=None意図的）。
    tech_winner はFoF合算後のSubB/SubCの多数決で決定される。
    差替時期: Phase4（精度93%のため低優先）
    """
    spy_12m = features.get('SPY_12m')
    is_quarter_start = ts.month in (1, 4, 7, 10)
    if is_quarter_start and spy_12m is not None and spy_12m < 0.06:
        return SubOutput(tech=1, choice=None, retreat=None)   # ATK
    else:
        return SubOutput(tech=0, choice=None, retreat='XLU')  # DEF


# ══════════════════════════════════════════════════════════════════
# Sub-B proxy: XLU守備 簡易版（Phase3でXLU52週高値距離に差替）
# ══════════════════════════════════════════════════════════════════
def subB_proxy(features: dict) -> SubOutput:
    """
    変数: TQQQ_3m（QQQ_3mではない）[修正済]
    placeholder: 本実装はPhase3。精度限定的。
    差替時期: Phase3（XLU 52週高値距離 閾値-0.15）
    """
    xlu_6m  = features.get('XLU_6m')
    tqqq_3m = features.get('TQQQ_3m')   # TQQQ_3m（QQQ_3mではない）
    xlu_healthy  = (xlu_6m  is not None and xlu_6m  > 0)
    tech_falling = (tqqq_3m is not None and tqqq_3m <= 0)
    if xlu_healthy and tech_falling:
        return SubOutput(tech=0, choice=None, retreat='XLU')
    else:
        return SubOutput(tech=1, choice='TQQQ', retreat=None)


# ══════════════════════════════════════════════════════════════════
# Sub-C proxy: credit + tech 簡易版（Phase4でKalman近似に差替）
# ══════════════════════════════════════════════════════════════════
def subC_proxy(features: dict) -> SubOutput:
    """
    変数: rel_TQ_12m（rel_TQ_6mではない）[修正済]
    差替時期: Phase4（Kalman近似 λ=0.94 EMA）
    """
    lqd_6m     = features.get('LQD_6m')
    rel_tq_12m = features.get('rel_TQ_12m')   # 12M（6Mではない）
    if lqd_6m is not None and lqd_6m > 0:
        choice = 'TECL' if (rel_tq_12m is not None and rel_tq_12m > 0) else 'TQQQ'
        return SubOutput(tech=1, choice=choice, retreat=None)
    else:
        return SubOutput(tech=0, choice=None, retreat='GLD')


# ══════════════════════════════════════════════════════════════════
# FoF合算ロジック
# ══════════════════════════════════════════════════════════════════
def compute_fof(sA: SubOutput, sB: SubOutput, sC: SubOutput, features: dict) -> dict:
    """3-Sub EqualWeight FoF合算。tie-break明示 [FIX-3]"""
    tech_v = sA.tech + sB.tech + sC.tech   # ∈ {0,1,2,3}
    def_v  = 3 - tech_v

    # tech選択（TECL vs TQQQ）
    tech_choices = [s.choice for s in [sA, sB, sC]
                    if s.tech == 1 and s.choice is not None]
    if not tech_choices:
        rel_tq = features.get('rel_TQ_12m')
        tech_winner = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
    else:
        counts = Counter(tech_choices); top2 = counts.most_common(2)
        if len(top2) >= 2 and top2[0][1] == top2[1][1]:
            # [FIX-3] tie-break: rel_TQ_12m > 0.07 → TECL
            rel_tq = features.get('rel_TQ_12m')
            tech_winner = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
        else:
            tech_winner = top2[0][0]

    # retreat選択（XLU vs GLD）
    retreats = [s.retreat for s in [sA, sB, sC]
                if s.tech == 0 and s.retreat is not None]
    if not retreats:
        retreat_winner = 'XLU'
    else:
        counts = Counter(retreats); top2 = counts.most_common(2)
        if len(top2) >= 2 and top2[0][1] == top2[1][1]:
            # [FIX-3] tie-break: LQD_6m > 0.02 → GLD
            lqd_6m = features.get('LQD_6m')
            retreat_winner = 'GLD' if (lqd_6m is not None and lqd_6m > 0.02) else 'XLU'
        else:
            retreat_winner = top2[0][0]

    # 配分計算（1/3単位・正規化）
    alloc_raw = {}
    if tech_v > 0:  alloc_raw[tech_winner]    = tech_v / 3 * 100
    if def_v  > 0:  alloc_raw[retreat_winner] = def_v  / 3 * 100
    total = sum(alloc_raw.values())
    alloc = ({k: round(v / total * 100, 1) for k, v in alloc_raw.items()}
             if total > 0 else {'XLU': 100.0})

    return {
        'alloc': alloc, 'gate': ('ATK' if tech_v >= 2 else 'DEF'),
        'tech_v': tech_v, 'def_v': def_v,
        'tech_winner': tech_winner, 'retreat_winner': retreat_winner,
    }


# ══════════════════════════════════════════════════════════════════
# yakuzai_v1 parallel monitor（§830.2完全移植）
# [INV-1] Subの入口ゲートとして使用禁止。FoF差分監視専用。
# ══════════════════════════════════════════════════════════════════
def yakuzai_v1_monitor(features: dict) -> dict:
    """FoF aggregate との tech_v 差分監視のみ。制御には使わない。"""
    TQQQ_3m = features.get('TQQQ_3m') or 0
    VIX     = features.get('VIX')     or 20
    LQD_6m  = features.get('LQD_6m')  or 0
    XLU_6m  = features.get('XLU_6m')  or 0
    TECL_6m = features.get('TECL_6m') or 0
    TQQQ_6m = features.get('TQQQ_6m') or 0
    GLD_3m  = features.get('GLD_3m')  or 0
    SPY_3m  = features.get('SPY_3m')  or 0   # [FIX-4]
    TECL_3m = features.get('TECL_3m') or 0   # [FIX-4]

    # §830.2 Offensive Gate
    if TQQQ_3m < -0.29:     def_v = 3 if XLU_6m > 0 else 2
    elif VIX > 26 and TQQQ_3m > -0.05: def_v = 0
    elif VIX <= 16:         def_v = 1 if LQD_6m > 0.04 else 2
    else:                   def_v = 2
    tech_v_pred = 3 - def_v

    # §829.4 GLD vs XLU split
    GLD_tech_3m  = GLD_3m - (TECL_3m + TQQQ_3m) / 2
    GLD_SPY_3m   = GLD_3m - SPY_3m
    TECL_TQQQ_6m = TECL_6m - TQQQ_6m
    if def_v == 0:
        gld_v = 0; xlu_v = 0
    else:
        gld_v = (1 if (XLU_6m > 0.12 and GLD_tech_3m > -0.15) or
                      (XLU_6m <= 0.12 and GLD_SPY_3m > 0.09) else 0)
        xlu_v = max(0, def_v - gld_v)

    # §830.2 TECL vs TQQQ split
    if tech_v_pred == 0:    tecl_v = 0; tqqq_v = 0
    elif TECL_TQQQ_6m > 0.02:
        tecl_v = min(tech_v_pred, 2); tqqq_v = tech_v_pred - tecl_v
    else:
        tqqq_v = min(tech_v_pred, 2); tecl_v = tech_v_pred - tqqq_v

    pred_alloc = {a: round(v/3, 4) for a, v in
                  {'GLD': gld_v, 'TECL': tecl_v, 'TQQQ': tqqq_v, 'XLU': xlu_v}.items()
                  if v > 0}
    return {'pred_tech_v': tech_v_pred, 'pred_def_v': def_v, 'pred_alloc': pred_alloc}


# ══════════════════════════════════════════════════════════════════
# compute_signals: 3-Sub FoF（戻り値スキーマ互換維持）
# ══════════════════════════════════════════════════════════════════
def compute_signals(f: dict) -> dict:
    ts = f.get('_ts') or pd.Timestamp(date.today().strftime('%Y-%m-01'))

    # 各Sub独立判断 [INV-2]
    sA = subA_proxy(f, ts)
    sB = subB_proxy(f)
    sC = subC_proxy(f)

    # FoF合算 [INV-3: 厳密1/3]
    fof = compute_fof(sA, sB, sC, f)

    # yakuzai_v1 parallel monitor [INV-1: 制御しない]
    monitor    = yakuzai_v1_monitor(f)
    tech_drift = abs(monitor['pred_tech_v'] - fof['tech_v'])

    gate  = fof['gate']
    alloc = fof['alloc']
    tech  = fof['tech_winner']
    lqd_ex = f.get('LQD_ex'); rel_tq = f.get('rel_TQ_12m')
    bnd_l  = (lqd_ex is not None and abs(lqd_ex) < 0.01)
    bnd_t  = (rel_tq is not None and abs(rel_tq) < 0.05)

    holdings = [
        {'ticker': tk, 'name': ETF_DESC.get(tk, tk),
         'weight': round(w / 100, 4), 'color': ETF_COLORS.get(tk, '#888')}
        for tk, w in alloc.items()
    ]
    return dict(
        gate=gate, sz=1.0, tech_choice=tech, has_xlu=('XLU' in alloc),
        boundary_LQD=bnd_l, boundary_tech=bnd_t, alert=bnd_l or bnd_t,
        alloc=alloc, holdings=holdings,
        layers={
            'A': gate,
            'B': f'SubB:{"ATK" if sB.tech else "DEF"}',
            'C': f'SubC:{"ATK" if sC.tech else "DEF"}',
            'D': tech,
        },
        fof_detail={
            'tech_v': fof['tech_v'], 'def_v': fof['def_v'],
            'subA': {'tech': sA.tech, 'retreat': sA.retreat, 'choice': sA.choice},
            'subB': {'tech': sB.tech, 'retreat': sB.retreat, 'choice': sB.choice},
            'subC': {'tech': sC.tech, 'retreat': sC.retreat, 'choice': sC.choice},
        },
        yakuzai_monitor={
            'pred_tech_v': monitor['pred_tech_v'],
            'actual_tech_v': fof['tech_v'],
            'drift': round(tech_drift, 3),
            'pred_alloc': monitor['pred_alloc'],
        },
    )


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    today  = date.today()
    target = pd.Timestamp(f'{today.year}-{today.month:02d}-01')
    label  = target.strftime('%Y-%m')
    logging.info(f'generate_signal_alpha.py v2.0 (3-Sub FoF) — {label}')

    prices, dtb3, hy, vix = load_data()
    f   = compute_features(target, prices, dtb3, hy, vix)
    sig = compute_signals(f)

    # _tsはJSONシリアライズ不可のため除外
    f_out = {k: v for k, v in f.items() if k != '_ts'}

    out = {
        'date':            label,
        'generated':       datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy':        'α',
        'engine':          '3-Sub EqualWeight FoF v2.0',
        'gate':            sig['gate'],
        'sz':              sig['sz'],
        'layers':          sig['layers'],
        'alloc':           sig['alloc'],
        'holdings':        sig['holdings'],
        'alert':           sig['alert'],
        'boundary_LQD':    sig['boundary_LQD'],
        'boundary_tech':   sig['boundary_tech'],
        'features':        f_out,
        'fof_detail':      sig['fof_detail'],
        'yakuzai_monitor': sig['yakuzai_monitor'],
    }

    for path in [OUTPUT / 'signal_etf_latest.json',
                 OUTPUT / f'signal_etf_{label.replace("-", "_")}.json']:
        with open(path, 'w') as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)

    logging.info(f'  Gate={sig["gate"]}  tech_v={sig["fof_detail"]["tech_v"]}  '
                 f'alloc={sig["alloc"]}  '
                 f'yakuzai_drift={sig["yakuzai_monitor"]["drift"]}')
    return out


if __name__ == '__main__':
    main()
