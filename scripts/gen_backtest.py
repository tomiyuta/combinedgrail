#!/usr/bin/env python3
"""
Combined Grail — バックテストデータ生成
HolyETF M4 (ret_M4) × OG防御型 の月次リターンを各配分比率で合成
ETF側は HolyGrail_ETF/output/cumulative_returns.json のM4データを使用（本家実績）
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

_CG_ROOT = Path(__file__).parent.parent
ETF_CUM  = Path('/Users/yutatomi/Downloads/01_投資・定量分析/HolyGrail_ETF/output/cumulative_returns.json')
OG_CUM   = Path('/Users/yutatomi/Downloads/01_投資・定量分析/HolyGrail_static/output/cumulative_returns.json')
OUTPUT   = _CG_ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)

def to_ym(d): return d[:7]

def stats(rets):
    r = pd.Series(rets, dtype=float).dropna()
    if len(r) < 6: return {}
    cagr    = float((1+r).prod()**(12/len(r))-1)
    sh      = float(r.mean()/r.std()*12**0.5) if r.std()>0 else 0
    # 半分散Sortino（holyetf.vercel.appと同一方式）
    semi    = np.where(r < 0, r, 0)
    ds      = float(np.sqrt(np.mean(semi**2)) * 12**0.5)
    sortino = float(r.mean()*12/ds) if ds>0 else 0
    cum=(1+r).cumprod(); md=float(((cum-cum.cummax())/cum.cummax()).min())
    calmar  = abs(cagr/md) if md!=0 else 0
    return dict(cagr=round(cagr,4), sharpe=round(sh,4),
                sortino=round(sortino,4), maxdd=round(md,4),
                calmar=round(calmar,4))

def cum_from_rets(rets):
    c = [1.0]
    for r in rets: c.append(round(c[-1]*(1+r), 6))
    return c[1:]

def annual_rets(dates, rets):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'r': rets})
    df['year'] = df['date'].dt.year
    result = {}
    for y, g in df.groupby('year'):
        result[str(y)] = round(float((1+pd.Series(g['r'].values)).prod()-1), 4)
    return result

def main():
    etf_data = json.load(open(ETF_CUM))
    og_data  = json.load(open(OG_CUM))

    # HolyETF M4 月次リターン
    etf_M4 = {d: r for d,r in zip(etf_data['dates'], etf_data['ret_M4'])}
    # SPY (HolyETF側を使用)
    etf_SPY = {d: r for d,r in zip(etf_data['dates'], etf_data['ret_SPY'])}
    # OG防御型 月次リターン
    og_def  = {to_ym(d): r for d,r in zip(og_data['dates'], og_data['ret_defense'])}

    # 共通期間
    common = sorted(set(etf_M4.keys()) & set(og_def.keys()))
    print(f"共通期間: {common[0]} ~ {common[-1]} ({len(common)}ヶ月)")

    ret_etf = [etf_M4[d] for d in common]
    ret_def = [og_def[d]  for d in common]
    ret_spy = [etf_SPY.get(d, 0) for d in common]

    # 配分比率リスト
    ratios = [(0,100),(10,90),(20,80),(30,70),(40,60),(50,50),(60,40),(70,30),(80,20),(90,10),(100,0)]
    key_map = {(e,d): f'r{e}_{d}' for e,d in ratios}

    out = {'dates': common}

    # 各配分の累積・月次リターン・年次・統計
    ret_all = {}
    for e_pct, d_pct in ratios:
        e = e_pct/100; d = d_pct/100
        rets = [e*re + d*rd for re,rd in zip(ret_etf, ret_def)]
        key  = f'r{e_pct}_{d_pct}'
        out[key]          = cum_from_rets(rets)
        out[f'ret_{key}'] = [round(r,6) for r in rets]
        ret_all[key] = rets

    # SPY
    out['SPY']     = cum_from_rets(ret_spy)
    out['ret_SPY'] = [round(r,6) for r in ret_spy]

    # 統計
    out['stats'] = {k: stats(ret_all[k]) for k in ret_all}
    out['stats']['SPY'] = stats(ret_spy)

    # 年次リターン（全配分＋SPY）
    out['annual'] = {k: annual_rets(common, ret_all[k]) for k in ret_all}
    out['annual']['SPY'] = annual_rets(common, ret_spy)

    path = OUTPUT / 'cumulative_returns.json'
    with open(path, 'w') as f:
        json.dump(out, f)

    print("✅ cumulative_returns.json 生成完了")
    print(f"  ETF単体(r100_0): CAGR={out['stats']['r100_0']['cagr']:.1%} "
          f"Sharpe={out['stats']['r100_0']['sharpe']:.2f} "
          f"MaxDD={out['stats']['r100_0']['maxdd']:.1%}")
    print(f"  推奨(r40_60):    CAGR={out['stats']['r40_60']['cagr']:.1%} "
          f"Sharpe={out['stats']['r40_60']['sharpe']:.2f} "
          f"MaxDD={out['stats']['r40_60']['maxdd']:.1%}")
    print(f"  防御型(r0_100):  CAGR={out['stats']['r0_100']['cagr']:.1%} "
          f"Sharpe={out['stats']['r0_100']['sharpe']:.2f} "
          f"MaxDD={out['stats']['r0_100']['maxdd']:.1%}")

if __name__ == '__main__':
    main()
