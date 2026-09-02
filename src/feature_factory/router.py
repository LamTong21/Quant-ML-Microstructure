# src/feature_factory/router.py

import pandas as pd
from typing import Dict
from .base_strategy import BaseFeatureStrategy
from .microstructure import L2MicrostructureStrategy, OrderFlowToxicityStrategy, LiquidityMicrostructureStrategy, VWAPPressureStrategy
from .kinematics import KinematicDynamicsStrategy, TrendMomentumStrategy, BaselineStrategy
from .volatility_stress import VolatilityDynamicsStrategy, AsymmetricStressStrategy, VolatilityTermStructureStrategy, VolatilityJumpDiffusionStrategy
from .spectral_wavelets import WaveletMultiResolutionStrategy, FractionalMemoryStrategy, MultiScaleComplexityStrategy, NonlinearInteractionStrategy
from .candle_shadows import IntradayShadowPressureStrategy
from .regime_clustering import GMMRegimeStrategy, RegimeConditionedStrategy

class Layer10FeatureRouter:
    """
    Điểm truy cập điều hướng (Router-Driven).
    Tự động gắn kết các Feature Strategies dựa trên giấy phép phân tích từ Stage 2 Payload.
    """
    def __init__(self, payload: dict):
        self.payload = payload
        self.registry: Dict[str, BaseFeatureStrategy] = {}
        self._dispatch()

    def _dispatch(self):
        flags = self.payload.get('flags', {})
        lags = self.payload.get('dynamic_lags', [1, 3, 5, 10, 20])
        opt_d = self.payload.get('optimal_d', 1.0)

        # Baseline & Unconditional (Luôn khởi tạo)
        self.registry["Baseline & Geometry"] = BaselineStrategy(dynamic_lags=lags)
        self.registry["Liquidity Microstructure"] = LiquidityMicrostructureStrategy()
        self.registry["Kinematics & Flow"] = KinematicDynamicsStrategy()
        self.registry["Regime Dynamics"] = RegimeConditionedStrategy()
        self.registry["GMM State Clustering"] = GMMRegimeStrategy(window=120, update_freq=20)
        self.registry["Intraday Shadow Pressure"] = IntradayShadowPressureStrategy()
        self.registry["Wavelet Multi-Resolution"] = WaveletMultiResolutionStrategy()

        # Conditional Strategies (Route based on Stage 2 flags)
        if flags.get('has_long_trend', True):
            self.registry["Trend & Momentum"] = TrendMomentumStrategy(dynamic_lags=lags)

        if 0.0 < opt_d < 1.0:
            self.registry["Fractional Memory"] = FractionalMemoryStrategy(optimal_d=opt_d)

        if flags.get('has_vol_clustering', True):
            self.registry["Volatility Dynamics"] = VolatilityDynamicsStrategy(
                dynamic_lags=lags,
                has_asymmetric_vol=flags.get('has_asymmetric_vol', False)
            )
            self.registry["Volatility Term Structure"] = VolatilityTermStructureStrategy()

        if flags.get('has_asymmetric_vol', False):
            self.registry["Asymmetric Stress"] = AsymmetricStressStrategy(window=20)

        if flags.get('has_vol_jumps', False):
            self.registry["Volatility Jump Diffusion"] = VolatilityJumpDiffusionStrategy(window=20)

        if flags.get('is_high_complexity', False):
            self.registry["Multi-Scale Complexity"] = MultiScaleComplexityStrategy()

        if flags.get('is_nonlinear', True):
            self.registry["Nonlinear Interactions"] = NonlinearInteractionStrategy()

        # L2 Microstructure Hook
        if flags.get('has_l2_orderbook', False):
            self.registry["L2 Real Order Book"] = L2MicrostructureStrategy()
            self.registry["Order Flow Toxicity (L2)"] = OrderFlowToxicityStrategy()
        elif flags.get('is_volume_significant', True):
            self.registry["Order Flow Toxicity (Proxy)"] = OrderFlowToxicityStrategy()
            self.registry["VWAP Pressure"] = VWAPPressureStrategy(window=20)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frames = []
        for name, strategy in self.registry.items():
            mod_feat = strategy.construct(df)
            feature_frames.append(mod_feat)

        return pd.concat(feature_frames, axis=1)