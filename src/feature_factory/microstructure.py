# src/feature_factory/microstructure.py

import numpy as np
import pandas as pd
from .base_strategy import BaseFeatureStrategy

class L2MicrostructureStrategy(BaseFeatureStrategy):
    """Khai thác động lực học vi mô thực tế từ dữ liệu Sổ lệnh (L2 Order Book)."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if 'micro_mid_deviation' in df.columns:
            feat['l2_micro_dev_raw'] = df['micro_mid_deviation']
            feat['l2_micro_dev_accel'] = feat['l2_micro_dev_raw'] - feat['l2_micro_dev_raw'].shift(1)
            
            if 'l2_spread' in df.columns:
                spread = df['l2_spread']
                feat['l2_spread_zscore'] = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + eps)

            b_size = df.get('bid_size_1', df.get('bid_size', pd.Series(0, index=df.index)))
            a_size = df.get('ask_size_1', df.get('ask_size', pd.Series(0, index=df.index)))

            if not (b_size.sum() == 0 and a_size.sum() == 0):
                imbalance = b_size - a_size
                feat['l2_order_imbalance_raw'] = imbalance
                feat['l2_order_imbalance_z'] = (imbalance - imbalance.rolling(20).mean()) / (imbalance.rolling(20).std() + eps)
                feat['l2_depth_total'] = b_size + a_size
                feat['l2_depth_ratio'] = b_size / (a_size + eps)

                delta_b = b_size.diff().fillna(0)
                delta_a = a_size.diff().fillna(0)
                feat['l2_bid_replenishment'] = np.where(delta_b > 0, delta_b, 0)
                feat['l2_ask_replenishment'] = np.where(delta_a > 0, delta_a, 0)
                feat['l2_bid_withdrawal'] = np.where(delta_b < 0, -delta_b, 0)
                feat['l2_ask_withdrawal'] = np.where(delta_a < 0, -delta_a, 0)
        return feat

class OrderFlowToxicityStrategy(BaseFeatureStrategy):
    """Tính toán OFI và VPIN Proxy."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['bid_size_1', 'ask_size_1']):
            order_imbalance = df['bid_size_1'] - df['ask_size_1']
        elif all(col in df.columns for col in ['bid_size', 'ask_size']):
            order_imbalance = df['bid_size'] - df['ask_size']
        else:
            c, o, h, l = df['close_adj'], df['open_adj'], df['high_adj'], df['low_adj']
            v = df['volume']
            buy_pressure = (c - l) / (h - l + eps)
            sell_pressure = (h - c) / (h - l + eps)
            order_imbalance = (buy_pressure - sell_pressure) * v

        feat['flow_imbalance_proxy'] = order_imbalance
        feat['flow_imbalance_zscore'] = (order_imbalance - order_imbalance.rolling(20).mean()) / (order_imbalance.rolling(20).std() + eps)

        v_series = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        vol_bucket = v_series.rolling(10).sum()
        imbalance_bucket = order_imbalance.abs().rolling(10).sum()
        feat['flow_vpin_10'] = imbalance_bucket / (vol_bucket + eps)

        return feat

class LiquidityMicrostructureStrategy(BaseFeatureStrategy):
    """Chi phí giao dịch ngầm, độ lệch Corwin-Schultz, Amihud và Roll Measure."""
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df['high_adj'], df['low_adj'], df['close_adj']
        v = df['volume']
        r = df['log_return']

        h_prev, l_prev = h.shift(1), l.shift(1)
        gamma = (np.log(h / l) ** 2) + (np.log(h_prev / l_prev) ** 2)
        h_2d = np.maximum(h, h_prev)
        l_2d = np.minimum(l, l_prev)
        beta = np.log(h_2d / l_2d) ** 2
        den = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        feat['liq_spread_cs'] = spread.clip(lower=0.0)

        dollar_volume = c * v
        amihud_raw = r.abs() / dollar_volume.replace(0, np.nan)
        feat['liq_amihud_raw'] = amihud_raw
        feat['liq_amihud_z'] = (amihud_raw - amihud_raw.rolling(20).mean()) / (amihud_raw.rolling(20).std() + eps)

        def calc_autocorr_lag1(x):
            return pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0

        feat['liq_roll_measure_20'] = r.rolling(20).apply(calc_autocorr_lag1, raw=False)
        feat['liq_log_turnover'] = np.log(dollar_volume + eps)
        feat['liq_turnover_ratio_20'] = dollar_volume / (dollar_volume.shift(1).rolling(20).mean() + eps)
        return feat

class VWAPPressureStrategy(BaseFeatureStrategy):
    """Áp lực Dòng lệnh và Hồ sơ Khối lượng (VWAP Profile)."""
    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df['high_adj'], df['low_adj'], df['close_adj']
        v = df['volume']

        tp = (h + l + c) / 3.0
        tp_v = tp * v
        roll_tp_v = tp_v.rolling(self.window).sum()
        roll_v = v.rolling(self.window).sum()
        vwap = roll_tp_v / (roll_v + eps)
        feat['vwap_distance_20'] = (c - vwap) / (vwap + eps)
        return feat