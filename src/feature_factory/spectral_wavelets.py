# src/feature_factory/spectral_wavelets.py

import numpy as np
import pandas as pd
from scipy.stats import entropy
from .base_strategy import BaseFeatureStrategy

class WaveletMultiResolutionStrategy(BaseFeatureStrategy):
    """Phân rã Đa phân giải Tần số Haar Wavelet Details."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']

        feat['wavelet_detail_d1'] = (p - p.shift(1)) / np.sqrt(2.0)
        feat['wavelet_detail_d2'] = ((p + p.shift(1)) - (p.shift(2) + p.shift(3))) / 2.0
        feat['wavelet_approx_a3'] = p.rolling(8).mean()
        feat['wavelet_energy_ratio'] = (feat['wavelet_detail_d1'] ** 2) / ((feat['wavelet_detail_d2'] ** 2) + eps)
        return feat

class FractionalMemoryStrategy(BaseFeatureStrategy):
    """Tích hợp vi phân phân số lưu giữ ký ức dài hạn của chuỗi."""
    def __init__(self, optimal_d: float, threshold: float = 1e-4, max_window: int = 100):
        self.d = optimal_d
        self.threshold = threshold
        self.max_window = max_window
        self.weights_forward = self._compute_weights(max_window)
        self.actual_max_len = len(self.weights_forward)

    def _compute_weights(self, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w)

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        
        def apply_frac_diff(x):
            x_rev = x[::-1]
            valid_len = min(len(x_rev), self.actual_max_len)
            return np.dot(self.weights_forward[:valid_len], x_rev[:valid_len])

        feat[f'frac_diff_d{self.d:.2f}'] = p.rolling(self.max_window, min_periods=10).apply(apply_frac_diff, raw=True)
        return feat

class MultiScaleComplexityStrategy(BaseFeatureStrategy):
    """Đo lường độ bất định thông qua Permutation Entropy."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return'].fillna(0)

        def calc_pe(x):
            if len(x) < 6: return np.nan
            m, tau = 3, 1
            n = len(x) - (m - 1) * tau
            patterns = np.array([x[i : i + m * tau : tau] for i in range(n)])
            ranks = np.argsort(patterns, axis=1)
            _, counts = np.unique(ranks, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(6.0)

        feat['chaos_permutation_entropy_20'] = r.rolling(20).apply(calc_pe, raw=True)
        binary_seq = (r > 0).astype(int)
        feat['lz_complexity_proxy_20'] = (binary_seq.diff().abs()).rolling(20).mean()
        return feat

class NonlinearInteractionStrategy(BaseFeatureStrategy):
    """Tương tác Phi tuyến giữa Lợi suất và Khối lượng."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return']
        h, l = df['high_adj'], df['low_adj']

        feat['nonlin_ret_squared_signed'] = (r ** 2) * np.sign(r)
        feat['nonlin_ret_cubed'] = r ** 3
        daily_range = np.log(h / l)
        feat['nonlin_range_x_ret'] = daily_range * r

        vol_roll_mean = df['volume'].shift(1).rolling(20).mean()
        feat['nonlin_ret_x_volshock'] = r * (df['volume'] / (vol_roll_mean + eps))
        return feat