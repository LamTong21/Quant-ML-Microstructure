# src/dgp_diagnostics/scanner.py

import pandas as pd
from .memory_spectral import MemoryDependence, SpectralCycleDiagnostics
from .fractional_diff import FractionalIntegration
from .volatility_jumps import VolatilityDynamics, VolatilityJumpDiagnostics
from .complexity_chaos import NonlinearDependence, ComplexityDiagnostics
from .volume_microstructure import VolumeDiagnostics, MicrostructureDiagnostics

class DGPScanner:
    @staticmethod
    def execute(df: pd.DataFrame) -> dict:
        returns = df['log_return'].dropna()
        prices = df['close_adj'].dropna()

        mem_stats = MemoryDependence.multi_scale_memory(prices, returns)
        cycle_stats = SpectralCycleDiagnostics.run(prices)
        frac_stats = FractionalIntegration.find_optimal_d(prices)
        vol_stats = VolatilityDynamics.estimate(df)
        jump_stats = VolatilityJumpDiagnostics.run(df)
        volu_stats = VolumeDiagnostics.run(df)
        nonlin_stats = NonlinearDependence.run(returns)
        complex_stats = ComplexityDiagnostics.run(returns)
        micro_stats = MicrostructureDiagnostics.run(df)

        routing_payload = {
            "dynamic_lags": mem_stats['dynamic_lags'],
            "optimal_d": frac_stats['optimal_d'],
            "dominant_cycle": cycle_stats['dominant_cycle_len'],
            "flags": {
                "has_long_trend": mem_stats['long_term_trending'],
                "has_short_reversion": mem_stats['short_term_mean_reverting'],
                "has_vol_clustering": vol_stats['has_arch_effect'],
                "has_asymmetric_vol": vol_stats['has_asymmetric_vol'],
                "is_volume_significant": volu_stats['is_volume_significant'],
                "is_nonlinear": nonlin_stats['is_nonlinear_dependent'],
                "has_vol_jumps": jump_stats['has_volatility_jumps'],
                "is_high_complexity": complex_stats['is_high_complexity'],
                "has_l2_orderbook": micro_stats['has_l2_orderbook'],
                "micro_deviation_sig": micro_stats['micro_deviation_sig']
            }
        }
        return routing_payload