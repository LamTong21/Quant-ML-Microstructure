# src/feature_factory/kinematics.py

import numpy as np
import pandas as pd
from .base_strategy import BaseFeatureStrategy

class KinematicDynamicsStrategy(BaseFeatureStrategy):
    """Động lực học Động lượng: Velocity, Acceleration, Jerk và Squeeze Ratio."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        h, l = df['high_adj'], df['low_adj']
        v = df['volume']
        r = df['log_return'].fillna(0)

        feat['kinematic_velocity'] = r
        feat['kinematic_acceleration'] = r - r.shift(1)
        feat['kinematic_jerk'] = feat['kinematic_acceleration'] - feat['kinematic_acceleration'].shift(1)

        roll_std = p.shift(1).rolling(20).std(ddof=1)
        bb_width = 4.0 * roll_std

        p_prev = p.shift(1)
        tr1 = h - l
        tr2 = (h - p_prev).abs()
        tr3 = (l - p_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.shift(1).rolling(20).mean()
        kc_width = 3.0 * atr

        feat['kinematic_squeeze_ratio'] = bb_width / (kc_width + eps)

        clv = ((p - l) - (h - p)) / (h - l + eps)
        feat['kinematic_clv_vol'] = clv * v
        feat['kinematic_clv_vol_roll20'] = feat['kinematic_clv_vol'].rolling(20).mean()
        return feat

class TrendMomentumStrategy(BaseFeatureStrategy):
    """Đo lường động lượng tuyến tính đa quy mô."""
    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']

        for k in self.lags:
            ret_k = np.log(p / p.shift(k))
            feat[f'ret_{k}'] = ret_k
            feat[f'tsmom_sign_{k}'] = np.sign(ret_k)
            if k >= 3:
                roll_max = p.shift(1).rolling(k).max()
                roll_min = p.shift(1).rolling(k).min()
                feat[f'breakout_ratio_{k}'] = (p - roll_max) / ((roll_max - roll_min) + eps)

        ema_12 = p.ewm(span=12, adjust=False).mean()
        ema_26 = p.ewm(span=26, adjust=False).mean()
        feat['macd_dist'] = (ema_12 / ema_26) - 1.0
        feat['macd_slope_5'] = feat['macd_dist'] - feat['macd_dist'].shift(5)
        return feat

class BaselineStrategy(BaseFeatureStrategy):
    """Đặc trưng hình học cơ sở và Gap qua đêm."""
    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df['high_adj'], df['low_adj'], df['close_adj'], df['open_adj']
        v = df['volume']

        candle_range = (h - l) + eps
        feat['geo_body_ratio'] = (c - o) / candle_range
        feat['geo_upper_shadow'] = (h - np.maximum(o, c)) / candle_range
        feat['geo_lower_shadow'] = (np.minimum(o, c) - l) / candle_range
        feat['overnight_gap'] = (o / c.shift(1)) - 1.0

        log_v = np.log(v + eps)
        for k in self.lags:
            if k >= 3:
                roll_vol_mean = v.shift(1).rolling(k).mean()
                feat[f'rvol_{k}'] = v / (roll_vol_mean + eps)
                mu_lv = log_v.shift(1).rolling(k).mean()
                std_lv = log_v.shift(1).rolling(k).std(ddof=1)
                feat[f'vol_shock_{k}'] = (log_v - mu_lv) / (std_lv + eps)

        if isinstance(df.index, pd.DatetimeIndex):
            dow = df.index.dayofweek
            feat['sin_dow'] = np.sin(2 * np.pi * dow / 5.0)
            feat['cos_dow'] = np.cos(2 * np.pi * dow / 5.0)
        return feat