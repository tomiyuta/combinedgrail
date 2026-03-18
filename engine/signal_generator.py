#!/usr/bin/env python3
"""
HolyGrail_ETF — Signal Generator  (Standalone)
Version: 1.9 | 2026-03-16 | Persistence ALPHA=0.15確定(WF ΔSh=+0.060 本採用✅)
  USE_B5_V2=True  → B5v2 (SOXL+XLV候補)  Sharpe+0.033 / OOS後半に劣後クラスター要監視
  USE_B5_V2=False → B5_current (旧Production)
  研究採用: ✅ / 本番固定: 保留 (OOS N≥12で再評価)
No external BAM dependency. All paths resolved within HolyGrail_ETF/.

Naming:
  Sensors:  CREDIT / RATES / ROTATION / MOMENTUM
  Modes:    FORTRESS / PIVOT / HYBRID / NEUTRAL / ADVANCE / SURGE
  Regimes:  CAUTIOUS / PREEMPTIVE / STRESS / ACUTE
"""
import os, sys, json, math, warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ── エンジンフォルダを sys.path に追加（data_loader を import するため）────
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_ENGINE_DIR)   # HolyGrail_ETF/
sys.path.insert(0, _ENGINE_DIR)

from data_loader import DataLoader, DATA_PATHS

# ════════════════════════════════════════════════════════════════
# FEATURE FLAG — P2 candidate
# ════════════════════════════════════════════════════════════════
# True  = B5v2 (SOXL+XLV): 研究採用済み / 本番固定は保留 / OOS後半劣後要監視
# False = B5_current: 旧Production / 比較ベースライン維持
USE_B5_V2 = True

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
ASSETS = ['GLD', 'XLU', 'TMV', 'TECL', 'TQQQ', 'SOXL', 'XLV']

# ── B5_current (旧Production / ベースライン) ──────────────────────
_DISCRETE_B5_CURRENT = {
    'CAUTIOUS':             {'TQQQ':0.40,'TECL':0.30,'XLU':0.20,'GLD':0.10,'TMV':0.00},
    'PREEMPTIVE':           {'XLU':0.40,'GLD':0.20,'TQQQ':0.20,'TECL':0.10,'TMV':0.10},
    'STRESS':               {'GLD':0.30,'TMV':0.30,'XLU':0.30,'TQQQ':0.05,'TECL':0.05},
    'ACUTE':                {'GLD':0.30,'TMV':0.40,'XLU':0.30,'TQQQ':0.00,'TECL':0.00},
    'ACUTE_EQUITY':         {'TQQQ':0.35,'TECL':0.25,'GLD':0.25,'XLU':0.15,'TMV':0.00},
    'RATES_UP_HV_RECOVERY': {'TQQQ':0.40,'TECL':0.30,'GLD':0.20,'XLU':0.10,'TMV':0.00},
    'RATES_UP_HV_INFLATION':{'GLD':0.40,'XLU':0.25,'TLT':0.20,'XLV':0.15,'TMV':0.00},
    'RATES_UP_LOW_VOL':     {'TMV':0.40,'GLD':0.20,'XLU':0.20,'TQQQ':0.10,'TECL':0.10},
}

# ── B5v2 (P2 candidate / SOXL+XLV追加) ──────────────────────────
# 最終採用判断 (2026-03-16):
#   M1: 条件付き採用 (Aggressive sleeve / Sharpe+0.004 / 年率+4.8pp)
#   M3: 不採用寄り (Sharpe-0.001 / ボラ増が相殺)
#   M4: 現行維持確定 (Sharpe-0.003 / 守備設計と逆行)
# SOXLの本質: risk-on amplification tool (CAUTIOUS+0.88%/月 / PREEMPTIVE+0.43%/月)
#             system-wide improvement tool ではない
# CAUTIOUS: SOXL15% semiconductor beta
# PREEMPTIVE: XLV15% defensive growth (N=13 / LOO 13/13安定)
_DISCRETE_B5_V2 = {
    'CAUTIOUS':             {'TQQQ':0.35,'TECL':0.20,'SOXL':0.15,'XLU':0.20,'GLD':0.10,'TMV':0.00},
    'PREEMPTIVE':           {'XLU':0.25,'XLV':0.15,'GLD':0.15,'TQQQ':0.25,'TECL':0.10,'TMV':0.10},
    'STRESS':               {'GLD':0.30,'TMV':0.30,'XLU':0.30,'TQQQ':0.05,'TECL':0.05},
    'ACUTE':                {'GLD':0.30,'TMV':0.40,'XLU':0.30,'TQQQ':0.00,'TECL':0.00},
    'ACUTE_EQUITY':         {'TQQQ':0.35,'TECL':0.25,'GLD':0.25,'XLU':0.15,'TMV':0.00},
    'RATES_UP_HV_RECOVERY': {'TQQQ':0.40,'TECL':0.30,'GLD':0.20,'XLU':0.10,'TMV':0.00},
    'RATES_UP_HV_INFLATION':{'GLD':0.40,'XLU':0.25,'TLT':0.20,'XLV':0.15,'TMV':0.00},
    'RATES_UP_LOW_VOL':     {'TMV':0.40,'GLD':0.20,'XLU':0.20,'TQQQ':0.10,'TECL':0.10},
}

DISCRETE = _DISCRETE_B5_V2 if USE_B5_V2 else _DISCRETE_B5_CURRENT
logging.info(f"allocation model = {'B5_v2 (SOXL+XLV candidate)' if USE_B5_V2 else 'B5_current (baseline)'}")

# ════════════════════════════════════════════════════════════════
# CONVEXITY BUDGET — 4bucket ontology (SESSION 15 / 2026-03-16)
# ════════════════════════════════════════════════════════════════
# equity_convexity proxy (実測値): SOXL=1.589 / TECL=1.526 / TQQQ=1.391
# rates次元 (TLT基準): TMV=-0.834 (inverse→負), TLT=+0.400
# ⚠️ TMVはSPY/TLT基準proxyで負値 = "RATES_INVERSE" (rates-up局面でのみhedge機能)
CONVEXITY_BUCKET = {
    "SOXL": "EQUITY_HIGH",   # equity proxy 1.589
    "TECL": "EQUITY_HIGH",   # equity proxy 1.526
    "TQQQ": "EQUITY_HIGH",   # equity proxy 1.391
    "XLV":  "LOW",           # equity proxy 0.286
    "GLD":  "LOW",           # equity proxy 0.130
    "XLU":  "LOW",           # equity proxy 0.098
    "TMV":  "RATES_INVERSE", # equity proxy -0.058 / rates -0.834 / rates-up局面でhedge
    "TLT":  "RATES_LONG",    # rates proxy +0.400 / duration long
}

STATE_CONVEXITY_TARGET = {
    "CAUTIOUS":             {"equity": "HIGH",        "rates": "NONE"},
    "PREEMPTIVE":           {"equity": "MEDIUM_LOW",  "rates": "NONE"},
    "STRESS":               {"equity": "LOW",         "rates": "NONE"},
    "ACUTE_EQUITY":         {"equity": "HIGH",        "rates": "NONE"},
    "RATES_UP_LOW_VOL":     {"equity": "LOW",         "rates": "INVERSE_HIGH"},
    "RATES_UP_HV_RECOVERY": {"equity": "MEDIUM_HIGH", "rates": "MEDIUM"},
    "RATES_UP_HV_INFLATION":{"equity": "LOW",         "rates": "MIXED"},
}

# ════════════════════════════════════════════════════════════════
# LAYER 4.5 — SECTOR DISCONNECT MONITOR (SESSION 15 / 2026-03-16)
# P2-2結果: SOXL crash asymmetryなし → 真のリスクはsector idiosyncratic disconnect
# disconnect = SPY>0 AND (SOXL-TECL) ≤ -8%
# ════════════════════════════════════════════════════════════════
def detect_soxl_disconnect(spy_ret: float, soxl_ret: float, tecl_ret: float) -> bool:
    """
    SPY上昇中のSOXL sector disconnectを検出
    Doc15修正: SPY > 0 → SPY ≥ +1%（ノイズ除去 / SPY=+0.1%等の微小上昇は除外）
    """
    if spy_ret < 0.01:   # SPY<+1%はFalse positive除外（下落 + 微小上昇）
        return False
    return (soxl_ret - tecl_ret) <= -0.08

def apply_disconnect_monitor(alloc: dict, disconnect_streak: int) -> dict:
    """
    Layer4.5: sector disconnect streakに応じてSOXL weightを動的補正
    補完分はTQQQ/TECLに均等配分 → CAUTIOUS HIGH bucket ≥ 50%維持
    disconnect_streak: 0=Normal / 1=Warning / ≥2=Critical
    """
    if 'SOXL' not in alloc or disconnect_streak == 0:
        return alloc
    result = dict(alloc)
    base_w = alloc['SOXL']
    new_w  = min(base_w, 0.10 if disconnect_streak >= 2 else 0.12)
    delta  = base_w - new_w
    if delta <= 0:
        return result
    result['SOXL']  = new_w
    result['TQQQ']  = result.get('TQQQ', 0) + delta / 2
    result['TECL']  = result.get('TECL', 0) + delta / 2
    logging.warning(f"[Layer4.5] SOXL disconnect streak={disconnect_streak}: "
                    f"SOXL {base_w:.0%}→{new_w:.0%} / TQQQ+{delta/2:.1%} TECL+{delta/2:.1%}")
    return result
# ════════════════════════════════════════════════════════════════
# PERSISTENCE LAYER — BAM知見応用 (SESSION 15末 / 2026-03-16)
# BAM実測: prev_state importance=0.604 / accuracy +5.2pp
# HolyGrail実装: EMA特徴量平滑化方式（feature score bias方式は不適）
# 実測: α=0.10でSharpe+0.019 / Flips-5回(11.4%削減) / CAUTIOUS+3ヶ月
# 採用基準(+0.03)は未達だが方向性正しい / SESSION16でportfolio test実施
# ════════════════════════════════════════════════════════════════
# EMA persistence用の状態変数（運用時は外部から管理）
_ema_state = {
    'LQD_SHY_6m': None, 'VIX': None, 'TLT_6m': None,
    'SPY_6m': None,     'TLT_3m': None, 'SPY_3m': None,
}
PERSISTENCE_ALPHA = 0.15  # WF Grid Search確定値: ΔSh=+0.060 / Flip削減10% / 本採用✅ (Doc20)

def update_ema_state(row_data: dict, alpha: float = PERSISTENCE_ALPHA) -> dict:
    """EMA平滑化された特徴量を更新・返す（月次呼び出し）"""
    smoothed = {}
    for key in _ema_state:
        val = row_data.get(key, 0) or 0
        if _ema_state[key] is None:
            _ema_state[key] = val
        else:
            _ema_state[key] = (1 - alpha) * val + alpha * _ema_state[key]
        smoothed[key] = _ema_state[key]
    return smoothed

def reset_ema_state():
    """EMA状態をリセット（バックテスト開始時に使用）"""
    for k in _ema_state:
        _ema_state[k] = None

M2_RATIO = {'CAUTIOUS':0.50,'PREEMPTIVE':0.70,'STRESS':0.50,
            'ACUTE':0.80,'ACUTE_EQUITY':0.70,
            'RATES_UP_HV_RECOVERY':0.60,'RATES_UP_HV_INFLATION':0.50,
            'RATES_UP_LOW_VOL':0.60}

# ════════════════════════════════════════════════════════════════
# SPM LAYER — Shock Probability Model v1.2 統合
# ChatGPT Doc23仕様: state→prior→SPM補正→final budget→ETF weights
# USE_SPM=True でアクティベート / False で無効（Persistence+B5v2のみ）
# ════════════════════════════════════════════════════════════════
USE_SPM = False  # SESSION16 P1: True/Falseで比較評価

# State別 prior shock budget（§917確定値）
SPM_PRIOR_BUDGET = {
    'CAUTIOUS':             {'eq_h':0.65,'low':0.05,'rt':0.10},
    'PREEMPTIVE':           {'eq_h':0.30,'low':0.30,'rt':0.15},
    'STRESS':               {'eq_h':0.10,'low':0.40,'rt':0.25},
    'ACUTE_EQUITY':         {'eq_h':0.60,'low':0.10,'rt':0.10},
    'RATES_UP_LOW_VOL':     {'eq_h':0.15,'low':0.25,'rt':0.45},
    'RATES_UP_HV_RECOVERY': {'eq_h':0.50,'low':0.20,'rt':0.15},
    'RATES_UP_HV_INFLATION':{'eq_h':0.10,'low':0.30,'rt':0.35},
}
# SPM v1.2 baseline priors（全期間平均 / SPM_V12_OOS_PREDICTIONS.csvから）
SPM_BASELINE_ACC = 0.228   # y_acc mean（q80 threshold = 3.94%）
SPM_BASELINE_RT  = 0.244   # y_rt mean（TLT abs q80 = 4.44%）
SPM_LAMBDA       = 0.85    # prior dominance (λ=0.85確定値)

# SPM予測値の動的ロード（SPM_V12_OOS_PREDICTIONS.csvがあれば使用）
_spm_predictions = None

def _load_spm_predictions():
    """SPM v1.2 walk-forward予測値をロード（初回のみ）"""
    global _spm_predictions
    if _spm_predictions is not None:
        return _spm_predictions
    csv_path = os.path.join(os.path.dirname(_ROOT_DIR), 'BAM',
                            'SPM_V12_OOS_PREDICTIONS.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(str).str[:7]  # YYYY-MM形式に統一
        _spm_predictions = df
        logging.info(f"SPM v1.2 predictions loaded: N={len(df)}")
    else:
        logging.warning("SPM predictions not found, using state prior only")
        _spm_predictions = pd.DataFrame()
    return _spm_predictions

def _apply_spm_correction(state: str, date_str: str, weights: dict) -> dict:
    """
    SPM v1.2 budget補正 → ETF weights に反映
    フロー: state → prior_budget → SPM補正 → final_budget → weights scaling

    Args:
        state:    7状態レジーム名
        date_str: 'YYYY-MM' 形式の月
        weights:  compute_weights()で得たETF weight dict（合計≈1.0）

    Returns:
        SPM補正後のweights dict（合計1.0に正規化済み）
    """
    if not USE_SPM:
        return weights

    prior = SPM_PRIOR_BUDGET.get(state, {'eq_h':0.30,'low':0.30,'rt':0.15})

    # SPM予測値を取得（WF OOS予測が使えればそちらを優先、なければstate prior）
    preds = _load_spm_predictions()
    if not preds.empty and date_str[:7] in preds.index:
        row    = preds.loc[date_str[:7]]
        p_acc  = float(row.get('p_acc',  SPM_BASELINE_ACC))
        p_rt   = float(row.get('p_rt',   SPM_BASELINE_RT))
    else:
        p_acc  = SPM_BASELINE_ACC
        p_rt   = SPM_BASELINE_RT

    # model budget（仕様準拠: ベースラインとの差分のみで補正）
    # p_acc/p_rt がbaseline以上なら eq_h/rt を増やす方向、以下なら減らす
    # 最大補正幅を±0.10に制限してoverride防止
    max_delta = 0.10
    delta_eq = max(-max_delta, min(max_delta, (p_acc - SPM_BASELINE_ACC) * 0.30))
    delta_rt = max(-max_delta, min(max_delta, (p_rt  - SPM_BASELINE_RT)  * 0.30))

    model = {
        'eq_h': max(0.0, min(1.0, prior['eq_h'] + delta_eq)),
        'rt':   max(0.0, min(1.0, prior['rt']   + delta_rt)),
        'low':  max(0.0, 1.0 - max(0.0, prior['eq_h']+delta_eq) - max(0.0, prior['rt']+delta_rt)),
    }

    # λ=0.85 prior dominant混合
    final = {k: SPM_LAMBDA * prior[k] + (1 - SPM_LAMBDA) * model[k]
             for k in ['eq_h','rt','low']}

    # STRESS安全弁: eq_hの上昇を+5%に制限（Doc23推奨）
    if state == 'STRESS':
        final['eq_h'] = min(final['eq_h'], prior['eq_h'] + 0.05)

    # ETF weights をバケット単位でスケール
    eq_h_etfs = [e for e in weights if CONVEXITY_BUCKET.get(e) == 'EQUITY_HIGH']
    low_etfs  = [e for e in weights if CONVEXITY_BUCKET.get(e) == 'LOW']
    rt_etfs   = [e for e in weights if CONVEXITY_BUCKET.get(e) in ('RATES_INVERSE','RATES_LONG')]

    cur_eq_h = sum(weights.get(e, 0) for e in eq_h_etfs)
    cur_low  = sum(weights.get(e, 0) for e in low_etfs)
    cur_rt   = sum(weights.get(e, 0) for e in rt_etfs)

    new_w = dict(weights)
    # 各バケットを final budget 比率でスケール（元がゼロなら変えない）
    for e in eq_h_etfs:
        if cur_eq_h > 1e-6:
            new_w[e] = weights[e] * final['eq_h'] / cur_eq_h
    for e in low_etfs:
        if cur_low > 1e-6:
            new_w[e] = weights[e] * final['low'] / cur_low
    for e in rt_etfs:
        if cur_rt > 1e-6:
            new_w[e] = weights[e] * final['rt'] / cur_rt

    # 合計=1に正規化
    total = sum(new_w.values())
    if total > 1e-6:
        new_w = {k: round(v / total, 4) for k, v in new_w.items()}

    return new_w

# ════════════════════════════════════════════════════════════════
# LAYER 0 — REGIME GATE
# ════════════════════════════════════════════════════════════════
def regime_gate(VIX, LQD_SHY_6m, SPY_6m, TLT_3m=0.0, SPY_3m=0.0) -> str:
    """
    4 base regimes + ACUTE 3-way split + RATES_UP_LOW_VOL (v2 / 2026-03-16)

    ACUTE分岐ルール（2軸4象限フレームワーク）:
      VIX>28 AND TLT_3m<-0.02:
        SPY_3m>0 AND LQD_SHY_6m>0 → RATES_UP_HV_RECOVERY  (2009型)
        else                        → RATES_UP_HV_INFLATION  (2022型)
      VIX>28 AND TLT_3m≥-0.02      → ACUTE_EQUITY           (株式クラッシュ)
    RATES_UP_LOW_VOL: VIX≤28 AND TLT_3m<-0.02 (通常相場の金利上昇 / TMV機能する)
    """
    rates_up = (TLT_3m < -0.02)
    hi_vol   = (VIX > 28)

    if hi_vol:
        if rates_up:
            if SPY_3m > 0 and LQD_SHY_6m > 0:
                return 'RATES_UP_HV_RECOVERY'
            return 'RATES_UP_HV_INFLATION'
        return 'ACUTE_EQUITY'

    if rates_up:
        return 'RATES_UP_LOW_VOL'

    if LQD_SHY_6m > 0 and SPY_6m > 0:     return 'CAUTIOUS'
    if LQD_SHY_6m <= 0 and SPY_6m > 0:    return 'PREEMPTIVE'
    return 'STRESS'

# ════════════════════════════════════════════════════════════════
# LAYER 1 — SENSOR GATES
# ════════════════════════════════════════════════════════════════
def credit_sensor(LQD_SHY_6m) -> str:
    return 'ATK' if LQD_SHY_6m > 0.02 else 'DEF'

def rates_sensor(TLT_6m, TLT_3m) -> str:
    if TLT_6m > 0.03:  return 'ATK'
    if TLT_6m < -0.01: return 'DEF'
    return 'NEU'

def rotation_sensor(XLU_SPY_3m, XLU_6m) -> str:
    if XLU_SPY_3m > 0.05: return 'DEF'
    if XLU_6m > -0.01:    return 'NEU'
    return 'ATK'

def momentum_sensor(SPY_6m) -> str:
    return 'ATK' if SPY_6m > 0 else 'DEF'

# ════════════════════════════════════════════════════════════════
# LAYER 2 — MODE ROUTER
# ════════════════════════════════════════════════════════════════
def mode_router(credit, rates, rotation, TQQQ_3m) -> str:
    if credit == 'DEF' and rates == 'DEF':  return 'FORTRESS'
    if credit == 'DEF' and rates == 'ATK':  return 'PIVOT'
    if credit == 'ATK' and rates == 'ATK':
        if rotation == 'DEF':              return 'HYBRID'
        if TQQQ_3m > 0.15:               return 'SURGE'
        if rotation == 'NEU':             return 'NEUTRAL'
        return 'ADVANCE'
    return 'NEUTRAL'

# ════════════════════════════════════════════════════════════════
# SOFTMAX ROUTER
# ════════════════════════════════════════════════════════════════
def softmax(scores: dict) -> dict:
    vals = np.array(list(scores.values())); vals = vals - vals.max()
    exps = np.exp(vals)
    return {k: float(v/exps.sum()) for k,v in zip(scores.keys(), exps)}

def softmax_weights(regime, data, b_sk=0.0, b_sl=0.0) -> dict:
    t12=data.get('TQQQ_12m',0); te12=data.get('TECL_12m',0)
    xlu=data.get('XLU_6m',0);   gld=data.get('GLD_6m',0); tlt=data.get('TLT_3m',0)
    if regime=='CAUTIOUS':
        sc={'TQQQ':t12+0.1,'TECL':te12+0.1,'XLU':xlu,'GLD':gld-0.1,'TMV':-0.3}
    elif regime=='PREEMPTIVE':
        sc={'XLU':xlu+0.1,'GLD':gld+0.05,'TQQQ':t12-0.05,'TECL':te12-0.1,'TMV':tlt*(-2)}
    elif regime=='STRESS':
        sc={'GLD':gld+0.1,'TMV':tlt*(-2)+0.05,'XLU':xlu+0.05,'TQQQ':-0.2,'TECL':-0.2}
    else:
        sc={'GLD':gld+0.1,'TMV':tlt*(-2)+0.1,'XLU':xlu,'TQQQ':-0.3,'TECL':-0.3}
    xsk=data.get('SKEW_norm',0); xsl=data.get('slope_norm',0)
    for a in sc:
        sc[a] += (b_sk*xsk+b_sl*xsl) if a in ['GLD','TMV'] else -(b_sk*xsk+b_sl*xsl)*0.5
    return softmax(sc)

# ════════════════════════════════════════════════════════════════
# LAYER 3 — C5 OVERLAY
# ════════════════════════════════════════════════════════════════
def c5_check(data) -> bool:
    return (abs(data.get('SKEW_norm',0))>0.5 or abs(data.get('slope_norm',0))>0.5) \
           and data.get('TLT_3m',0) >= -0.02

def rates_acute(data) -> bool:
    return data.get('TLT_3m',0) < -0.02

def compute_weights(regime, data):
    d=DISCRETE[regime]; s1=softmax_weights(regime,data); s2=softmax_weights(regime,data,-0.3,-0.3)
    # SOXL/XLV/TLT等の新ETFはsoftmax未対応 → discrete-only (s1[a]が無ければd[a]をそのまま使用)
    core = [a for a in ASSETS if a in s1]
    ext  = [a for a in d.keys() if a not in s1]
    w1={a:round(0.5*d.get(a,0)+0.5*s1[a],4) for a in core}
    w1.update({a:d.get(a,0) for a in ext})
    w2={a:round(0.5*d.get(a,0)+0.5*s2[a],4) for a in core}
    w2.update({a:d.get(a,0) for a in ext})
    # ── 正規化: SOXL/XLV/TLT追加でsoftmax合計が1.0超になる問題を修正 ──
    def _normalize(w):
        t = sum(w.values())
        return {k: round(v/t, 4) for k,v in w.items()} if t > 1e-6 else w
    w1 = _normalize(w1); w2 = _normalize(w2)
    c5=c5_check(data); ra=rates_acute(data)
    w3={a:round(0.5*w1[a]+0.5*w2[a],4) for a in w1} if c5 else dict(w1)
    w3 = _normalize(w3)
    lam=M2_RATIO.get(regime,0.3) if c5 else 0.0
    if ra: lam=min(lam,0.2)
    w4={a:round((1-lam)*w1[a]+lam*w2[a],4) for a in w1}
    w4 = _normalize(w4)
    return w1, w3, w4

# ════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════
def generate_signal(data: dict) -> dict:
    reg  = regime_gate(data['VIX'], data['LQD_SHY_6m'], data['SPY_6m'],
                       data.get('TLT_3m', 0.0), data.get('SPY_3m', 0.0))
    cr   = credit_sensor(data['LQD_SHY_6m'])
    rt   = rates_sensor(data['TLT_6m'], data['TLT_3m'])
    ro   = rotation_sensor(data.get('XLU_SPY_3m',0), data.get('XLU_6m',0))
    mo   = momentum_sensor(data['SPY_6m'])
    mode = mode_router(cr, rt, ro, data.get('TQQQ_3m',0))
    c5=c5_check(data); ra=rates_acute(data)
    w1,w3,w4=compute_weights(reg,data)
    return {'regime':reg,'sensors':{'CREDIT':cr,'RATES':rt,'ROTATION':ro,'MOMENTUM':mo},
            'mode':mode,'c5_active':c5,'rates_acute':ra,
            'weights_M1':w1,'weights_M3':w3,'weights_M4':w4}

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    dl = DataLoader(); dl.load()
    # ── 特徴量構築 ────────────────────────────────────────────────
    g = pd.DataFrame(index=pd.period_range('2016-01','2026-03',freq='M'))
    for tk,w in [('LQD',6),('LQD',3),('BIL',6),('BIL',3),('TLT',3),('TLT',6),
                 ('SPY',6),('SPY',3),('TQQQ',3),('TQQQ',12),('XLU',6),('GLD',6),('TECL',12)]:
        try: g[f'{tk}_{w}m'] = dl.get_rolling_return(tk, w)
        except: pass
    g['VIX'] = dl.vix
    # SHYの代替にBIL(T-bill ETF)を使用 / Gate C = LQD超過リターン vs 短期債
    g['LQD_SHY_6m'] = g['LQD_6m'] - g.get('BIL_6m', pd.Series(0,index=g.index))
    g['LQD_SHY_3m'] = g['LQD_3m'] - g.get('BIL_3m', pd.Series(0,index=g.index))
    g['XLU_SPY_3m'] = g.get('XLU_6m', pd.Series(0,index=g.index)) - \
                      g.get('SPY_6m',  pd.Series(0,index=g.index))

    # ── C5 vol指標 (HolyGrail_ETF/data内のCSVを直接参照) ─────────
    _phase_c = DATA_PATHS['phase_c']
    ki = pd.read_csv(_phase_c, index_col=0, parse_dates=True)
    ki.index = ki.index.to_period('M')
    ki['SKEW_norm']  = (ki['SKEW']       - ki['SKEW'].mean())       / ki['SKEW'].std()
    ki['slope_norm'] = (ki['term_slope'] - ki['term_slope'].mean()) / ki['term_slope'].std()
    g = g.join(ki[['SKEW_norm','slope_norm']], how='left')
    g = g.dropna(subset=['LQD_SHY_6m','VIX','TLT_6m','SPY_6m']).sort_index()

    # ── 最新月シグナル ─────────────────────────────────────────────
    latest = g.iloc[-1]
    sig = generate_signal(latest.to_dict())
    sig['date'] = str(g.index[-1])

    print("="*60)
    print(f"HolyGrail_ETF Signal — {sig['date']}")
    print("="*60)
    print(f"  Regime:   {sig['regime']}")
    s = sig['sensors']
    print(f"  Sensors:  CREDIT={s['CREDIT']}  RATES={s['RATES']}  "
          f"ROTATION={s['ROTATION']}  MOMENTUM={s['MOMENTUM']}")
    print(f"  Mode:     {sig['mode']}")
    print(f"  C5:       {'ACTIVE ★' if sig['c5_active'] else 'OFF'}"
          f"  | Rates-acute: {'YES' if sig['rates_acute'] else 'NO'}")
    print()
    for label,key in [('M1 (Conservative)','weights_M1'),
                      ('M3 (Recommended)', 'weights_M3'),
                      ('M4 (Aggressive)',  'weights_M4')]:
        w = sig[key]
        print(f"  {label}:")
        print(f"    GLD={w['GLD']:.1%}  XLU={w['XLU']:.1%}  "
              f"TMV={w['TMV']:.1%}  TECL={w['TECL']:.1%}  TQQQ={w['TQQQ']:.1%}")

    # ── JSON保存 ──────────────────────────────────────────────────
    out_dir = os.path.join(_ROOT_DIR, 'output')
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"signal_{sig['date'].replace('-','_')}.json")
    with open(fname, 'w') as f: json.dump(sig, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {fname}")
