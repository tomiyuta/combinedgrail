#!/usr/bin/env python3
"""CombinedGrail — αシグナル生成 (3-Sub EqualWeight FoF v3.0)
設計書: ALPHA_FoF_DESIGN_v1.6

=== Phase4-current-best 最終凍結版 ===
Sub-A v3  : DM3型 4銘柄Top1 proxy（VIX/XLU_SPY_3m gate）
Sub-B     : XLU 52週高値距離本実装（閾値-0.15）
Sub-C     : LQD creditゲート + tech proxy（XLU退避）
retreat集約: 比例配分（A-fix v2）/ tech集約: winner-take-all
評価       : exact=31.1% / tech_v_acc=58.8%✅ / MAE=0.577

[INV-1] yakuzai_v1 は parallel monitor 専用。各Subの入口ゲートに使用禁止。
[INV-2] 3-Subは独立して判断する。
[INV-3] FoF合算は厳密1/3等重。
[INV-4] Phase1+2は3要素同時投入。単独評価禁止。
[FROZEN] Sub-A/B/C実装・集約ロジックは変更禁止。GLD gate改善は研究ブランチで別途実施。
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

FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='

ETF_DESC = {
    'TECL': 'テクノロジー 3倍レバレッジ', 'TQQQ': 'NASDAQ 3倍レバレッジ',
    'XLU':  '公益セクター（守備）',        'GLD':  '金・クライシスヘッジ',
    'GDX':  '金鉱株ETF（守備）',           'TMV':  '長期国債 逆3倍（金利上昇ヘッジ）',
}
ETF_COLORS = {
    'TECL': '#F59E0B', 'TQQQ': '#60A5FA', 'XLU': '#34D399',
    'GLD':  '#D4AF7A', 'GDX':  '#B45309', 'TMV': '#A78BFA',
}
PRICE_TICKERS = ['LQD', 'TECL', 'TQQQ', 'XLU', 'GLD', 'TMV', 'SPY']

# ─────────────────────────────────────────────────────────────────
# Sub共通インターフェース
# ─────────────────────────────────────────────────────────────────
@dataclass
class SubOutput:
    tech:    int            # 0=守備 / 1=攻撃（1Subあたり1票）
    choice:  Optional[str]  # 'TECL' or 'TQQQ'（tech=1時）
    retreat: Optional[str]  # 'XLU' or 'GLD'（tech=0時）


# ─────────────────────────────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────────────────────────────
def fetch_fred(series_id: str) -> pd.Series:
    url = FRED_BASE + series_id
    logging.info(f'  FRED {series_id} 取得中...')
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                     na_values='.').iloc[:, 0].dropna()
    return df.resample('MS').last()

def fetch_prices(tickers, years: int = 2) -> pd.DataFrame:
    start = (date.today().replace(day=1) - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    logging.info(f'  yfinance {tickers} 取得中... (start={start})')
    raw = yf.download(tickers, start=start, interval='1mo',
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (raw['Close'] if 'Close' in raw.columns.get_level_values(0)
                  else raw.xs('Close', axis=1, level=1))
    else:
        prices = raw
    prices.index = prices.index.to_period('M').to_timestamp()
    return prices

def load_data():
    """Phase1+2追加: SPY, VIX / Phase3追加: XLU日次（52週高値距離）"""
    prices = fetch_prices(PRICE_TICKERS)
    dtb3   = fetch_fred('DTB3') / 100
    hy     = fetch_fred('BAMLH0A0HYM2')
    vix    = fetch_fred('VXVCLS')   # VIX追加 [Phase1+2]

    # Phase3: XLU 52週高値距離（norgate日次データ）
    xlu_dist_series = _build_xlu_dist()

    return prices, dtb3, hy, vix, xlu_dist_series


def _build_xlu_dist() -> pd.Series:
    """XLU 52週高値距離を日次で計算して返す（norgate parquet使用）"""
    NORGATE = Path('/Users/yutatomi/Downloads/01_投資・定量分析/norgate_unified.parquet')
    if not NORGATE.exists():
        logging.warning('norgate_unified.parquet が見つからない。XLU_52w_dist=Noneで動作')
        return pd.Series(dtype=float)
    df = pd.read_parquet(NORGATE, columns=['XLU_AdjClose'])
    xlu = df['XLU_AdjClose'].dropna()
    high52 = xlu.rolling(252).max()
    return (xlu - high52) / high52


# ─────────────────────────────────────────────────────────────────
# 特徴量計算（compute_features）
# ─────────────────────────────────────────────────────────────────
def compute_features(ts: pd.Timestamp, prices, dtb3, hy, vix,
                     xlu_dist_series: pd.Series = None) -> dict:
    """
    ts: 当月（シグナル適用月）
    prev: 前月末（シグナル算出タイミング）
    全数値判定は is not None で統一 [FIX-1]
    """
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
    # [FIX-1] is not None で統一（0.0・負値で偽判定しない）
    lqd_ex  = (lqd_12m - dtb_v) if (lqd_12m is not None and dtb_v is not None) else None
    hy_v    = float(hy.loc[prev]) if prev in hy.index else None

    tecl_12m = ret('TECL', 12)
    tqqq_12m = ret('TQQQ', 12)
    rel_tq_12m = ((tecl_12m - tqqq_12m)
                  if (tecl_12m is not None and tqqq_12m is not None) else None)

    # Phase1+2 追加特徴量
    vix_v    = float(vix.loc[prev]) if prev in vix.index else None
    spy_12m  = ret('SPY', 12)
    spy_3m   = ret('SPY', 3)    # yakuzai_v1 GLD_SPY_3m用 [FIX-4]
    lqd_6m   = ret('LQD', 6)
    tqqq_3m  = ret('TQQQ', 3)
    xlu_3m   = ret('XLU', 3)
    xlu_6m   = ret('XLU', 6)
    gld_3m   = ret('GLD', 3)
    gld_6m   = ret('GLD', 6)
    tecl_6m  = ret('TECL', 6)
    tecl_3m  = ret('TECL', 3)   # yakuzai_v1 GLD_tech_3m用 [FIX-4]
    tqqq_6m  = ret('TQQQ', 6)
    rel_tq_6m = ((tecl_6m - tqqq_6m)
                 if (tecl_6m is not None and tqqq_6m is not None) else None)

    return {
        # 既存（互換維持）
        'LQD_ex':        lqd_ex,
        'LQD_12m':       lqd_12m,
        'rel_TQ_12m':    rel_tq_12m,
        'mom4m_TMV':     ret('TMV', 4),
        'XLU_12m':       ret('XLU', 12),
        'HY_spread_pct': hy_v,
        # Phase1+2 追加
        'VIX':           vix_v,
        'SPY_12m':       spy_12m,
        'SPY_3m':        spy_3m,
        'LQD_6m':        lqd_6m,
        'TQQQ_3m':       tqqq_3m,
        'XLU_3m':        xlu_3m,
        'XLU_6m':        xlu_6m,
        'GLD_3m':        gld_3m,
        'GLD_6m':        gld_6m,
        'TECL_6m':       tecl_6m,
        'TECL_3m':       tecl_3m,
        'TQQQ_6m':       tqqq_6m,
        'rel_TQ_6m':     rel_tq_6m,
        # Phase3: Sub-B用 XLU 52週高値距離
        'XLU_52w_dist':  _get_xlu_dist(ts, xlu_dist_series) if xlu_dist_series is not None else None,
        # Sub-A v3用: GLD gate（VIX + XLU_SPY_3m）
        'XLU_SPY_3m':    ((xlu_3m - spy_3m)
                          if (xlu_3m is not None and spy_3m is not None) else None),
        # Sub-A用（四半期判定のためtsを埋め込む）
        '_ts':           ts,
    }


def _get_xlu_dist(ts: pd.Timestamp, xlu_dist_series: pd.Series) -> Optional[float]:
    """前月末のXLU 52週高値距離を取得"""
    if xlu_dist_series is None or len(xlu_dist_series) == 0:
        return None
    pm_end = ts - pd.DateOffset(days=1)
    avail = xlu_dist_series.loc[:pm_end.strftime('%Y-%m-%d')]
    return float(avail.iloc[-1]) if len(avail) > 0 else None


# ─────────────────────────────────────────────────────────────────
# Sub-A v3: 4銘柄 Top1 DM proxy（KB deep research確定版）
# ─────────────────────────────────────────────────────────────────
def subA_proxy(f: dict) -> SubOutput:
    """
    Sub-A(212e9eee = DM3) v3: 4銘柄Top1モメンタム選択proxy
    ユニバース: {TECL, GLD, XLU, TQQQ}

    KB確定（§25753/§25865/§27951）:
      212e9eee = DM3 (GLD/Swing)。月次Top1選択。
      実測頻度: TECL=47.5%, GLD=34.2%, XLU=11.7%, TQQQ=6.7%
      GLD選択条件（KB §51806/§71539）:
        VIX > 15.9 AND XLU_SPY_3m > -0.02 → GLD（acc=64.7%, LOOCV=51.8%）
        最重要特徴: VIX(importance=0.701)

    ロジック:
      1) GLD gate（唯一の既知シグナル）
      2) XLU候補（xlu_6m > 0.10）
      3) TECL/TQQQ（rel_TQ_12m で切り替え）

    [撤回] §5788 GLD50/TECL25/TQQQ25 固定配分仮説（特殊月の観測であり一般則ではない）
    [撤回] §494 四半期逆張り proxy（DM3は月次Top1選択DM）
    """
    vix        = f.get('VIX')
    xlu_spy_3m = f.get('XLU_SPY_3m')  # XLU_3m - SPY_3m [features追加済み]
    rel_tq_12m = f.get('rel_TQ_12m')
    xlu_6m     = f.get('XLU_6m')

    # 1) GLD候補（VIX > 15.9 AND XLU_SPY_3m > -0.02）
    if (vix is not None and xlu_spy_3m is not None
            and vix > 15.9 and xlu_spy_3m > -0.02):
        return SubOutput(tech=0, choice=None, retreat='GLD')

    # 2) XLU候補（6Mリターン > 10%: 実測11.7%に相応の頻度を確保）
    if xlu_6m is not None and xlu_6m > 0.10:
        return SubOutput(tech=0, choice=None, retreat='XLU')

    # 3) TECL / TQQQ（rel_TQ_12m による相対選択）
    if rel_tq_12m is not None and rel_tq_12m > 0:
        return SubOutput(tech=1, choice='TECL', retreat=None)
    else:
        return SubOutput(tech=1, choice='TQQQ', retreat=None)


# ─────────────────────────────────────────────────────────────────
# Sub-B proxy: XLU守備 簡易版（Phase3でXLU 52週高値距離に差替）
# ─────────────────────────────────────────────────────────────────
def subB_proxy(f: dict) -> SubOutput:
    """
    Sub-B(8650d48d) 本実装（Phase3）
    XLU 52週高値距離 >= -0.15 → XLU守備
    XLU 52週高値距離 <  -0.15 → TQQQ攻撃

    KB確定: Sub-B = XLU88%+TQQQ12%のほぼ守備専用Sub
    データ: norgate XLU日次終値から252営業日ローリング最大値で計算
    距離 = (現値 - 52週高値) / 52週高値  ← 前月末の値を使用

    [NOTE] XLU_52w_dist は compute_features で前月末に計算・格納される
    """
    dist = f.get('XLU_52w_dist')

    if dist is None or dist >= -0.15:
        return SubOutput(tech=0, choice=None,   retreat='XLU')  # XLU守備
    else:
        return SubOutput(tech=1, choice='TQQQ', retreat=None)   # TQQQ攻撃


# ─────────────────────────────────────────────────────────────────
# Sub-C proxy: credit + tech 簡易版（Phase4でKalman近似に差替）
# ─────────────────────────────────────────────────────────────────
def subC_proxy(f: dict) -> SubOutput:
    """
    Sub-C(c7477396) 簡易proxy
    LQD_6m > 0 → ATK: rel_TQ_12m > 0 → TECL else TQQQ
    LQD_6m <= 0 → DEF: XLU退避

    [修正] DEF退避先 GLD→XLU（KB確定: c7477396 = LQD creditゲート + TECL/TQQQ momentum）
    GLDはSub-Aの固定混合（GLD50%）由来。SubCがGLDを出す必要はない。
    [NOTE] Phase4でKalman近似（λ=0.94 EMA）に差替予定
    [NOTE] rel_TQ_12m を使用（6Mではなく12M統一）[FIX: v1.1]
    """
    lqd_6m     = f.get('LQD_6m')
    rel_tq_12m = f.get('rel_TQ_12m')

    if lqd_6m is not None and lqd_6m > 0:
        choice = 'TECL' if (rel_tq_12m is not None and rel_tq_12m > 0) else 'TQQQ'
        return SubOutput(tech=1, choice=choice, retreat=None)
    else:
        return SubOutput(tech=0, choice=None, retreat='XLU')  # XLU（GLD不可）


# ─────────────────────────────────────────────────────────────────
# FoF合算（compute_fof）
# ─────────────────────────────────────────────────────────────────
def compute_fof(sA: SubOutput, sB: SubOutput, sC: SubOutput, f: dict) -> dict:
    """
    3-Sub EqualWeight FoF合算
    tech_v + def_v = 3 の対角行列制約維持
    tie-break を明示 [FIX-3]
    """
    tech_v = sA.tech + sB.tech + sC.tech   # ∈ {0,1,2,3}
    def_v  = 3 - tech_v

    # ── tech選択（TECL vs TQQQ）tie-break明示 [FIX-3] ──
    tech_choices = [s.choice for s in (sA, sB, sC)
                    if s.tech == 1 and s.choice is not None]
    if not tech_choices:
        rel_tq = f.get('rel_TQ_12m')
        tech_winner = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
    else:
        cnt = Counter(tech_choices)
        top = cnt.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            # tie-break: rel_TQ_12m > 0.07 → TECL
            rel_tq = f.get('rel_TQ_12m')
            tech_winner = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
        else:
            tech_winner = top[0][0]

    # ── retreat集計（比例配分）[A-fix v2] ──
    # equal-weight FoF の正しい実装: winner-take-all ではなく各Subの退避先を比例合算
    # 例: sA=GLD, sB=XLU, sC=XLU → GLD:1/3 + XLU:2/3
    # 根拠: Sub-Aが GLD を出しているのは固有判断であり、多数決で上書きするのは設計意図に反する
    retreat_votes = {}
    for s in (sA, sB, sC):
        if s.tech == 0 and s.retreat is not None:
            retreat_votes[s.retreat] = retreat_votes.get(s.retreat, 0) + 1

    # 後続の raw_alloc 計算で使うために retreat_winner を保持（単一コードパスに乗せる）
    # ただし実際の配分は個別退避先ごとに蓄積する（下のraw_alloc計算で上書き）
    retreat_winner = max(retreat_votes, key=retreat_votes.get) if retreat_votes else 'XLU'

    # ── 配分計算: tech比例 + retreat比例（1/3単位）+ 正規化 ──
    # tech側は winner-take-all（TECL vs TQQQ は単一銘柄選択）
    # retreat側は比例配分（Sub-A固有判断を保持）[A-fix v2]
    raw_alloc = {}
    if tech_v > 0:
        raw_alloc[tech_winner] = raw_alloc.get(tech_winner, 0) + tech_v / 3 * 100
    for asset, votes in retreat_votes.items():
        raw_alloc[asset] = raw_alloc.get(asset, 0) + votes / 3 * 100

    # 合計を100.0に正規化（丸め誤差対策）
    total = sum(raw_alloc.values())
    alloc = {k: round(v / total * 100, 1) for k, v in raw_alloc.items()} if total > 0 else {'XLU': 100.0}

    gate = 'ATK' if tech_v >= 2 else 'DEF'

    return {
        'alloc':          alloc,
        'gate':           gate,
        'tech_v':         tech_v,
        'def_v':          def_v,
        'tech_winner':    tech_winner,
        'retreat_winner': retreat_winner,
    }


# ─────────────────────────────────────────────────────────────────
# compute_fof_v2: Phase4 fixed-mix Sub対応FoF（設計書v1.3 補強①②）
# ─────────────────────────────────────────────────────────────────
def compute_fof_v2(sA: SubOutput, sB: SubOutput, sC: SubOutput, f: dict) -> dict:
    """
    Phase4 FoF合算ロジック
    [補強①] 合算順序:
      Step1: Sub-A(fixed_alloc) を先に 1/3スロットとして確定加算
             fixed_allocを多数決に混ぜない。1/3固定寄与として扱う。
      Step2: 残り2/3 を Sub-B / Sub-C の通常tech票ロジックで埋める
      Step3: fixed_contribution + dynamic を合算 → 正規化

    [補強②] gate再定義:
      Sub-AがFixed-mix Subのため、gateはSub-B/C 2票の有効多数決で決定
      gate = 'ATK' if (sB.tech + sC.tech) >= 1 else 'DEF'
    """
    # ── Step1: Sub-A fixed_alloc を 1/3スロット確定 ──
    fixed_contrib = {k: v / 3 for k, v in sA.fixed_alloc.items()}  # 各銘柄の1/3寄与

    # ── Step2: Sub-B / Sub-C 2票のtech集計 ──
    tv2 = sB.tech + sC.tech   # ∈ {0, 1, 2}
    dv2 = 2 - tv2

    # tech選択（Sub-B/C 2票のみ）
    tc2 = [s.choice for s in (sB, sC) if s.tech == 1 and s.choice is not None]
    if not tc2:
        rel_tq = f.get('rel_TQ_12m')
        tw = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
    else:
        cnt = Counter(tc2)
        top = cnt.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            rel_tq = f.get('rel_TQ_12m')
            tw = 'TECL' if (rel_tq is not None and rel_tq > 0.07) else 'TQQQ'
        else:
            tw = top[0][0]

    # retreat選択（Sub-B/C 2票のみ）
    rc2 = [s.retreat for s in (sB, sC) if s.tech == 0 and s.retreat is not None]
    if not rc2:
        rw = 'XLU'
    else:
        cnt = Counter(rc2)
        top = cnt.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            lqd_6m = f.get('LQD_6m')
            rw = 'GLD' if (lqd_6m is not None and lqd_6m > 0.02) else 'XLU'
        else:
            rw = top[0][0]

    # ── Step3: dynamic部分（Sub-B/C 2/3スロット）──
    dynamic_contrib = {}
    if tv2 > 0:
        dynamic_contrib[tw]  = tv2 / 3
    if dv2 > 0:
        dynamic_contrib[rw] = dynamic_contrib.get(rw, 0) + dv2 / 3

    # ── 合算: fixed(1/3) + dynamic(2/3) ──
    raw = {}
    for k, v in fixed_contrib.items():
        raw[k] = raw.get(k, 0) + v
    for k, v in dynamic_contrib.items():
        raw[k] = raw.get(k, 0) + v

    total = sum(raw.values())
    alloc = {k: round(v / total * 100, 1) for k, v in raw.items() if v > 0} if total > 0 else {'XLU': 100.0}

    # ── [補強②] gate: Sub-B/C 2票の有効多数決 ──
    gate = 'ATK' if (sB.tech + sC.tech) >= 1 else 'DEF'

    return {
        'alloc':          alloc,
        'gate':           gate,
        'tech_v_bc':      tv2,          # Sub-B/C票のみ（Sub-A除く）
        'tech_winner':    tw,
        'retreat_winner': rw,
        'fixed_contrib':  {k: round(v, 4) for k, v in fixed_contrib.items()},
        'dynamic_contrib': {k: round(v, 4) for k, v in dynamic_contrib.items()},
    }


# ─────────────────────────────────────────────────────────────────
# yakuzai_v1 parallel monitor（§830.2 完全移植）
# [INV-1] FoF出口の差分監視専用。Subの入口ゲートとして使用禁止。
# ─────────────────────────────────────────────────────────────────
def yakuzai_v1_monitor(f: dict) -> dict:
    """
    §830.2 yakuzai_v1: FoF aggregate との乖離監視専用。
    制御には使わない。[INV-1]
    SPY_3m / TECL_3m は features に必須 [FIX-4]
    """
    TQQQ_3m  = f.get('TQQQ_3m') or 0.0
    VIX      = f.get('VIX')     or 20.0
    LQD_6m   = f.get('LQD_6m') or 0.0
    XLU_6m   = f.get('XLU_6m') or 0.0
    TECL_6m  = f.get('TECL_6m') or 0.0
    TQQQ_6m  = f.get('TQQQ_6m') or 0.0
    GLD_3m   = f.get('GLD_3m') or 0.0
    SPY_3m   = f.get('SPY_3m') or 0.0
    TECL_3m  = f.get('TECL_3m') or 0.0

    # §830.2 Offensive Gate: def_v (0-3)
    if TQQQ_3m < -0.29:
        def_v = 3 if XLU_6m > 0 else 2
    elif VIX > 26 and TQQQ_3m > -0.05:
        def_v = 0
    elif VIX <= 16:
        def_v = 1 if LQD_6m > 0.04 else 2
    else:
        def_v = 2
    tech_v = 3 - def_v

    # §829.4 GLD vs XLU split
    GLD_tech_3m  = GLD_3m - (TECL_3m + TQQQ_3m) / 2
    GLD_SPY_3m   = GLD_3m - SPY_3m
    TECL_TQQQ_6m = TECL_6m - TQQQ_6m

    if def_v == 0:
        gld_v = 0; xlu_v = 0
    else:
        gld_v = 1 if (
            (XLU_6m > 0.12 and GLD_tech_3m > -0.15) or
            (XLU_6m <= 0.12 and GLD_SPY_3m > 0.09)
        ) else 0
        xlu_v = max(0, def_v - gld_v)

    # §830.2 TECL vs TQQQ split
    if tech_v == 0:
        tecl_v = 0; tqqq_v = 0
    elif TECL_TQQQ_6m > 0.02:
        tecl_v = min(tech_v, 2); tqqq_v = tech_v - tecl_v
    else:
        tqqq_v = min(tech_v, 2); tecl_v = tech_v - tqqq_v

    pred_alloc = {a: v / 3 for a, v in
                  {'GLD': gld_v, 'TECL': tecl_v, 'TQQQ': tqqq_v, 'XLU': xlu_v}.items()
                  if v > 0}
    return {
        'pred_tech_v': tech_v,
        'pred_def_v':  def_v,
        'pred_alloc':  pred_alloc,
    }


# ─────────────────────────────────────────────────────────────────
# compute_signals: 3-Sub FoF メイン（旧単層エンジンを全面置換）
# ─────────────────────────────────────────────────────────────────
def compute_signals(f: dict) -> dict:
    """
    Phase3 + Sub-A v3: 3-Sub EqualWeight FoF
    Sub-A = DM3型 4銘柄Top1 proxy（v3）
    Sub-B = XLU 52週高値距離本実装（Phase3）
    Sub-C = LQD creditゲート + tech proxy（Phase3）
    FoF合算 = compute_fof（通常3票等重、設計書v1.3 Phase3ベース）
    """
    sA = subA_proxy(f)   # DM3型 Top1 proxy（v3）
    sB = subB_proxy(f)   # XLU 52週高値距離（Phase3本実装）
    sC = subC_proxy(f)   # credit+tech proxy（Phase3）

    fof = compute_fof(sA, sB, sC, f)   # Phase3ベース（通常3票等重）

    monitor    = yakuzai_v1_monitor(f)
    tech_drift = abs(monitor['pred_tech_v'] - fof['tech_v'])

    gate    = fof['gate']
    alloc   = fof['alloc']
    tech    = fof['tech_winner']
    retreat = fof['retreat_winner']

    lqd_ex  = f.get('LQD_ex')
    rel_tq  = f.get('rel_TQ_12m')
    bnd_l   = (lqd_ex is not None and abs(lqd_ex) < 0.01)
    bnd_t   = (rel_tq is not None and abs(rel_tq) < 0.05)

    holdings = [
        {'ticker': tk, 'name': ETF_DESC.get(tk, tk),
         'weight': round(w / 100, 4), 'color': ETF_COLORS.get(tk, '#888')}
        for tk, w in alloc.items()
    ]

    return dict(
        gate=gate, sz=1.0, tech_choice=tech, has_xlu=(retreat == 'XLU'),
        alloc=alloc, holdings=holdings,
        boundary_LQD=bnd_l, boundary_tech=bnd_t, alert=bnd_l or bnd_t,
        layers={
            'A': gate,
            'B': f'SubA:{"ATK" if sA.tech else "DEF"}({sA.choice or sA.retreat})',
            'C': f'SubB:{"ATK" if sB.tech else "DEF"}',
            'D': f'SubC:{"ATK" if sC.tech else "DEF"}',
        },
        fof_detail={
            'tech_v':  fof['tech_v'],
            'def_v':   fof['def_v'],
            'subA':    {'tech': sA.tech, 'choice': sA.choice, 'retreat': sA.retreat},
            'subB':    {'tech': sB.tech, 'retreat': sB.retreat, 'choice': sB.choice},
            'subC':    {'tech': sC.tech, 'retreat': sC.retreat, 'choice': sC.choice},
        },
        yakuzai_monitor={
            'pred_tech_v':   monitor['pred_tech_v'],
            'actual_tech_v': fof['tech_v'],
            'drift':         round(tech_drift, 3),
            'pred_alloc':    monitor['pred_alloc'],
        },
    )


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    today  = date.today()
    target = pd.Timestamp(f'{today.year}-{today.month:02d}-01')
    label  = target.strftime('%Y-%m')
    logging.info(f'generate_signal_alpha.py (3-Sub FoF v3.0 Sub-A v3) — {label}')

    prices, dtb3, hy, vix, xlu_dist = load_data()
    f   = compute_features(target, prices, dtb3, hy, vix, xlu_dist)
    sig = compute_signals(f)

    # features から _ts を除外（JSON非シリアライズ対象外）
    feat_out = {k: v for k, v in f.items()
                if k != '_ts' and not isinstance(v, pd.Timestamp)}

    out = {
        'date':           label,
        'generated':      datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'strategy':       'α',
        'engine_version': '3-Sub FoF v3.0 Sub-A v3',
        'gate':           sig['gate'],
        'sz':             sig['sz'],
        'layers':         sig['layers'],
        'alloc':          sig['alloc'],
        'holdings':       sig['holdings'],
        'alert':          sig['alert'],
        'boundary_LQD':   sig['boundary_LQD'],
        'boundary_tech':  sig['boundary_tech'],
        'features':       feat_out,
        'fof_detail':     sig['fof_detail'],
        'yakuzai_monitor': sig['yakuzai_monitor'],
    }

    for path in [OUTPUT / 'signal_etf_latest.json',
                 OUTPUT / f'signal_etf_{label.replace("-", "_")}.json']:
        with open(path, 'w') as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)

    logging.info(
        f'  Gate={sig["gate"]}  '
        f'tech_v={sig["fof_detail"]["tech_v"]}  '
        f'alloc={sig["alloc"]}  '
        f'yakuzai_drift={sig["yakuzai_monitor"]["drift"]}'
    )
    return out


if __name__ == '__main__':
    main()
