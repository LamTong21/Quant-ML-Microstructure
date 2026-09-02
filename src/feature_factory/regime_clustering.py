# src/feature_factory/regime_clustering.py

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from .base_strategy import BaseFeatureStrategy

class GMMRegimeStrategy(BaseFeatureStrategy):
    """Nhận diện Chế độ Rủi ro (GMM Bull/Bear Probabilities)."""
    def __init__(self, window: int = 120, update_freq: int = 20):
        self.window = window
        self.update_freq = update_freq

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return'].fillna(0)
        vol = r.rolling(20).std(ddof=1).fillna(0)

        n = len(df)
        gmm_bull = np.zeros(n)
        gmm_bear = np.zeros(n)
        X = np.column_stack((r.values, vol.values))

        for i in range(self.window, n, self.update_freq):
            start_idx = max(0, i - self.window)
            X_train = np.nan_to_num(X[start_idx:i])
            pred_end = min(n, i + self.update_freq)
            X_test = np.nan_to_num(X[i:pred_end])

            try:
                if np.var(X_train[:, 0]) > 1e-8:
                    gmm = GaussianMixture(n_components=2, random_state=42, n_init=1)
                    gmm.fit(X_train)
                    probs = gmm.predict_proba(X_test)
                    means = gmm.means_
                    
                    bull_idx = np.argmax(means[:, 0])
                    bear_idx = 1 - bull_idx
                    
                    gmm_bull[i:pred_end] = probs[:, bull_idx]
                    gmm_bear[i:pred_end] = probs[:, bear_idx]
            except Exception:
                pass

        feat['gmm_prob_bull'] = gmm_bull
        feat['gmm_prob_bear'] = gmm_bear
        feat.iloc[:self.window, :] = np.nan
        return feat

class RegimeConditionedStrategy(BaseFeatureStrategy):
    """Tính các biến phụ thuộc dựa trên rủi ro biến động cao/thấp."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return']

        roll_vol = r.rolling(20).std(ddof=1)
        vol_q50 = roll_vol.shift(1).rolling(60).median()
        p_high_vol = (roll_vol > vol_q50).astype(float)

        feat['regime_p_high_vol'] = p_high_vol
        feat['regime_weighted_return'] = r * p_high_vol
        feat['regime_transition_shock'] = p_high_vol.diff().abs()
        return feat