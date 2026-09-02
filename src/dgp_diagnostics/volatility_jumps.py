# src/dgp_diagnostics/volatility_jumps.py

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import het_arch

class VolatilityDynamics:
    @staticmethod
    def estimate(df: pd.DataFrame) -> dict:
        c, o = df['close_adj'], df['open_adj']
        ret = np.log(c / c.shift(1)).dropna()
        if len(ret) < 30:
            return {"has_arch_effect": False, "has_asymmetric_vol": False}

        lm_stat, lm_p, _, _ = het_arch(ret, nlags=5)
        
        ret_lag = ret.shift(1).dropna()
        vol_proxy = (ret**2).iloc[1:]
        aligned_df = pd.concat([ret_lag, vol_proxy], axis=1).dropna()
        leverage_corr = aligned_df.iloc[:,0].corr(aligned_df.iloc[:,1])

        return {
            "has_arch_effect": bool(lm_p < 0.05),
            "has_asymmetric_vol": bool(leverage_corr < -0.1)
        }

class VolatilityJumpDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame, window: int = 20) -> dict:
        r = df['log_return'].dropna()
        if len(r) < window * 2:
            return {"has_volatility_jumps": False}

        rv = (r ** 2).rolling(window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(window).sum()

        jump_ratio = np.maximum(rv - bv, 0) / (rv + 1e-8)
        significant_jumps = (jump_ratio > 0.25).sum()

        return {"has_volatility_jumps": bool(significant_jumps > (len(r) * 0.05))}