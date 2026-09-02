# src/feature_factory/candle_shadows.py

import numpy as np
import pandas as pd
from .base_strategy import BaseFeatureStrategy

class IntradayShadowPressureStrategy(BaseFeatureStrategy):
    """Áp lực Bóng nến Nội phiên: Tail Power và Shadow Asymmetry."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df['high_adj'], df['low_adj'], df['close_adj'], df['open_adj']
        v = df['volume']

        candle_range = (h - l) + eps
        body = (c - o).abs()
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        feat['shadow_asymmetry_ratio'] = (upper_shadow - lower_shadow) / candle_range
        feat['buying_tail_power'] = (lower_shadow / candle_range) * v
        feat['selling_tail_power'] = (upper_shadow / candle_range) * v
        feat['body_efficiency_ratio'] = body / candle_range
        return feat