# src/dgp_diagnostics/volume_microstructure.py

import pandas as pd
import numpy as np
from scipy import stats

class VolumeDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        v = df['volume'].dropna()
        r_abs = np.log(df['close_adj'] / df['close_adj'].shift(1)).abs().dropna()

        aligned = pd.concat([v, r_abs], axis=1).dropna()
        if aligned.empty or len(aligned) < 30:
            return {"is_volume_significant": False}

        corr, p_val = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        return {"is_volume_significant": bool(p_val < 0.05 and corr > 0.1)}

class MicrostructureDiagnostics:
    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        if 'micro_mid_deviation' not in df.columns:
            return {"has_l2_orderbook": False, "micro_deviation_sig": False}

        r = df['log_return'].fillna(0)
        micro_dev = df['micro_mid_deviation'].fillna(0)
        dev_corr, dev_pval = stats.spearmanr(micro_dev.shift(1).fillna(0), r)

        return {
            "has_l2_orderbook": True,
            "micro_deviation_sig": bool(dev_pval < 0.05 and abs(dev_corr) > 0.05)
        }