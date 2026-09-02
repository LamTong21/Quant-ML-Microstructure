# src/feature_selection/kinematic_lag_transform.py

import pandas as pd

class KinematicLagTransform:
    @staticmethod
    def apply_transform(X: pd.DataFrame, optimal_lags: dict, eps: float = 1e-8) -> pd.DataFrame:
        """
        Dịch chuyển chuỗi về đúng độ trễ nhân quả và tạo tính năng phái sinh (Momentum, Acceleration)[cite: 1, 2, 3].
        """
        X_transformed = pd.DataFrame(index=X.index)

        for col, lag in optimal_lags.items():
            if col not in X.columns:
                continue
            feature_series = X[col]

            # Dịch chuyển chuỗi dựa trên độ trễ nhân quả L*
            feature_lagged = feature_series.shift(lag)
            
            # 1. Z-Score Lag: Chuẩn hóa theo biến động 20 phiên
            roll_mean = feature_lagged.rolling(20).mean()
            roll_std = feature_lagged.rolling(20).std(ddof=1)
            z_scaled = (feature_lagged - roll_mean) / (roll_std + eps)
            X_transformed[f'{col}_zscaled_lag{lag}'] = z_scaled

            # 2. Kinematic Momentum (Động lượng bậc 1)
            momentum = (feature_series - feature_lagged) / (feature_lagged.abs() + eps)

            # 3. Kinematic Acceleration (Gia tốc bậc 2)
            acceleration = momentum - momentum.shift(1)

            X_transformed[f'{col}_momentum'] = momentum
            X_transformed[f'{col}_acceleration'] = acceleration

        return X_transformed