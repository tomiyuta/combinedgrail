#!/usr/bin/env python3
"""
Combined Grail — Data Updater
GitHub Actions の月次シグナル生成前に実行
prices_monthly・VIX を yfinance で最新月まで更新する
"""
import sys, warnings, logging
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

_CG_ROOT = Path(__file__).parent.parent
DATA_DIR  = _CG_ROOT / 'data'

PRICE_TICKERS = [
    'GLD','QLD','TMV','TECL','TLT','XLU','TQQQ','BIL',
    'GDX','LQD','QQQ','AGG','SPY','SOXL','XLV'
]

def update_prices_monthly():
    """prices_monthly_2004_2026.csv を最新月まで更新"""
    csv_path = DATA_DIR / 'market' / 'prices_monthly_2004_2026.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    last_date = df.index.max()
    logging.info(f"  現在の最終月: {last_date.date()}")

    today = date.today()
    # 先月末（現在月はまだ確定していないため）
    target = pd.Timestamp(today.year, today.month, 1) - pd.offsets.MonthEnd(1)
    if target <= last_date:
        logging.info(f"  更新不要（最新: {last_date.date()}）")
        return

    logging.info(f"  {last_date.date()} → {target.date()} を取得中...")
    start_str = (last_date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    end_str   = (target + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

    new_rows = {}
    for tk in PRICE_TICKERS:
        try:
            raw = yf.download(tk, start=start_str, end=end_str,
                              interval='1mo', auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if 'Close' in raw.columns and not raw.empty:
                for idx, row in raw.iterrows():
                    d = idx.to_timestamp() if hasattr(idx, 'to_timestamp') else idx
                    d = pd.Timestamp(d.year, d.month, 1)
                    if d not in new_rows: new_rows[d] = {}
                    new_rows[d][tk] = round(float(row['Close']), 10)
        except Exception as e:
            logging.warning(f"  {tk}: skip ({e})")

    if new_rows:
        new_df = pd.DataFrame(new_rows).T.sort_index()
        new_df.index.name = 'Date'
        df_updated = pd.concat([df, new_df[~new_df.index.isin(df.index)]])
        df_updated = df_updated.sort_index()
        df_updated.to_csv(csv_path)
        logging.info(f"  ✅ prices_monthly 更新: {len(new_rows)}行追加 → 最終月: {df_updated.index.max().date()}")
    else:
        logging.info("  価格データ取得なし")

def update_vix():
    """FRED_VXVCLS_auto.csv を最新月まで更新"""
    csv_path = DATA_DIR / 'fred' / 'FRED_VXVCLS_auto.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = ['VIX']
    last_date = df.index.max()
    logging.info(f"  VIX 現在の最終: {last_date.date()}")

    today = date.today()
    target = pd.Timestamp(today.year, today.month, 1) - pd.offsets.MonthEnd(0)
    if target <= last_date:
        logging.info(f"  VIX 更新不要")
        return

    logging.info(f"  VIX {last_date.date()} → {target.date()} 取得中...")
    start_str = last_date.strftime('%Y-%m-%d')
    try:
        raw = yf.download('^VIX', start=start_str,
                          interval='1mo', auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if 'Close' in raw.columns and not raw.empty:
            vix_m = raw['Close'].reset_index()
            vix_m.columns = ['Date', 'VIX']
            vix_m['Date'] = pd.to_datetime(vix_m['Date'])
            vix_m = vix_m.set_index('Date')
            # 月末に揃える
            vix_m.index = vix_m.index + pd.offsets.MonthEnd(0)
            new_rows = vix_m[vix_m.index > last_date]
            if not new_rows.empty:
                df_updated = pd.concat([df, new_rows[~new_rows.index.isin(df.index)]])
                df_updated = df_updated.sort_index()
                df_updated.columns = ['VIX']
                df_updated.index.name = 'Date'
                df_updated.to_csv(csv_path)
                logging.info(f"  ✅ VIX 更新: {len(new_rows)}行追加 → 最終: {df_updated.index.max().date()}")
    except Exception as e:
        logging.warning(f"  VIX 取得失敗: {e}")

def main():
    logging.info(f"Data Update — {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    update_prices_monthly()
    update_vix()
    logging.info("✅ データ更新完了")

if __name__ == '__main__':
    main()
