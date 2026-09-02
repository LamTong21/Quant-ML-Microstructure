# src/dgp_diagnostics/complexity_chaos.py

import pandas as pd
import numpy as np
import math
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import bds
from sklearn.feature_selection import mutual_info_regression

class NonlinearDependence:
    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 50:
            return {"is_nonlinear_dependent": False}

        ar_res = AutoReg(r.values, lags=1).fit()
        resid = ar_res.resid
        resid_std = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-8)

        bds_stat, p_val = bds(resid_std, max_dim=2, epsilon=None)

        return {"is_nonlinear_dependent": bool(p_val < 0.05)}

class ComplexityDiagnostics:
    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna().values
        if len(r) < 30: return {"is_high_complexity": False}

        m, tau = 3, 1
        n = len(r) - (m - 1) * tau
        patterns = np.array([r[i : i + m * tau : tau] for i in range(n)])
        ranks = np.argsort(patterns, axis=1)
        _, counts = np.unique(ranks, axis=0, return_counts=True)
        probs = counts / counts.sum()

        pe = -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(math.factorial(m))

        return {"is_high_complexity": bool(pe > 0.85)}