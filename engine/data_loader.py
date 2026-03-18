"""
HolyGrail_ETF - data_loader.py
Standalone version: paths fixed to HolyGrail_ETF/data/
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── CombinedGrail/data/ 内のファイルを参照（GitHub Actions対応） ──
_ENGINE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR   = _ENGINE_DIR.parent / 'data'

def _find_vix_file():
    """自動更新ファイルを優先し、なければ初期ファイルを使用"""
    auto  = _DATA_DIR / 'fred' / 'FRED_VXVCLS_auto.csv'
    orig  = _DATA_DIR / 'fred' / 'FRED_VXVCLS_20260306.csv'
    return str(auto) if auto.exists() else str(orig)

DATA_PATHS = {
    'prices':  str(_DATA_DIR / 'market' / 'prices_monthly_2004_2026.csv'),
    'vix':     _find_vix_file(),
    'dtb3':    str(_DATA_DIR / 'fred'   / 'FRED_DTB3_20260306.csv'),
    'phase_c': str(_DATA_DIR / 'phase_c' / 'phase_c_key_indicators_monthly.csv'),
}


class DataLoader:
    def __init__(self):
        self.etf  = None   # 月次リターン DataFrame (Period index)
        self.vix  = None   # VIX Series
        self.dtb3 = None   # 3M T-bill Series
        self._loaded = False

    def load(self):
        # ── ETF月次価格→リターン ──────────────────────────────────
        price = pd.read_csv(DATA_PATHS['prices'], index_col=0, parse_dates=True)
        price_m = price.resample('ME').last()
        price_m.index = price_m.index.to_period('M')
        self.etf = price_m.pct_change()   # 月次リターン

        # ── VIX ─────────────────────────────────────────────────
        vix = pd.read_csv(DATA_PATHS['vix'], index_col=0, parse_dates=True)
        vix.columns = ['VIX']
        vix = vix[vix['VIX'] != '.'].copy()
        vix['VIX'] = vix['VIX'].astype(float)
        vix_m = vix.resample('ME').mean()
        vix_m.index = vix_m.index.to_period('M')
        self.vix = vix_m['VIX']

        # ── DTB3 ─────────────────────────────────────────────────
        try:
            dtb = pd.read_csv(DATA_PATHS['dtb3'], index_col=0, parse_dates=True)
            dtb.columns = ['DTB3']
            dtb = dtb[dtb['DTB3'] != '.'].copy()
            dtb['DTB3'] = dtb['DTB3'].astype(float)
            dtb_m = dtb.resample('ME').mean()
            dtb_m.index = dtb_m.index.to_period('M')
            self.dtb3 = dtb_m['DTB3']
        except Exception:
            self.dtb3 = pd.Series(dtype=float)

        self._loaded = True
        rows = len(self.etf) if self.etf is not None else 0
        print(f"[DataLoader] ETF: {self.etf.shape}, VIX: {len(self.vix)}, DTB3: {len(self.dtb3)}")

    def get_rolling_return(self, ticker: str, months: int) -> pd.Series:
        if not self._loaded:
            self.load()
        if ticker not in self.etf.columns:
            raise KeyError(f"{ticker} not in ETF data")
        r = self.etf[ticker].dropna()
        return (1 + r).rolling(months).apply(lambda x: x.prod(), raw=True) - 1
