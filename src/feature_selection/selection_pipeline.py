# src/feature_selection/selection_pipeline.py

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .redundancy_control import RedundancyControl
from .causality_screen import CausalityScreen
from .kinematic_lag_transform import KinematicLagTransform
from .information_theory import InformationTheory

class SelectionPipelineRouter:
    """Pipeline điều phối đa tầng: Redundancy -> Causality -> Transformation -> Predictive Information[cite: 1, 3]."""
    def __init__(self, payload: dict):
        self.payload = payload
        self.max_clusters = payload.get('max_clusters', 15)
        self.mi_threshold = payload.get('mi_threshold', 0.01)

    def execute(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        aligned = pd.concat([X_train, y_train], axis=1).dropna()
        if aligned.empty:
            return X_train, {}

        X_mat = aligned.iloc[:, :-1]
        y_mat = aligned.iloc[:, -1]

        # PHASE 1: Khử dư thừa cấu trúc tuyến tính và phi tuyến
        print(f"  -> Lọc Tuyến tính VIF (Đầu vào: {X_mat.shape[1]} features)...")
        X_pruned = RedundancyControl.linear_vif_prune(X_mat, threshold=0.85, max_vif=5.0)

        print(f"  -> Lọc Phi tuyến HRP Spearman (Đầu vào: {X_pruned.shape[1]} features)...")
        X_hrp = RedundancyControl.nonlinear_hrp_prune(X_pruned, max_clusters=self.max_clusters)

        # Cứu lại cột trạng thái thị trường (regime) để dùng làm điều kiện cho MI
        if 'regime_p_high_vol' in X_train.columns and 'regime_p_high_vol' not in X_hrp.columns:
            X_hrp['regime_p_high_vol'] = X_train['regime_p_high_vol']

        # PHASE 2: Sàng lọc nhân quả kép
        print(f"  -> Sàng lọc Nhân quả (Granger & Transfer Entropy Proxy)...")
        valid_causal_feats, optimal_lags = CausalityScreen.screen(X_hrp, y_mat)

        # PHASE 3: Máy Biến đổi trễ & Động lực học
        print(f"  -> Biến đổi trễ & Động lực học (Lag Transformation)...")
        X_transformed = KinematicLagTransform.apply_transform(X_hrp, optimal_lags)

        # Thêm lại Regime flag vào ma trận đã biến đổi
        if 'regime_p_high_vol' in X_hrp.columns:
            X_transformed['regime_p_high_vol'] = X_hrp['regime_p_high_vol']

        # PHASE 4: Information Theory (Mutual Information có điều kiện Regime)
        print(f"  -> Khảo sát lý thuyết thông tin (Regime-Conditioned MI)...")
        valid_features = []
        X_transformed_clean = X_transformed.dropna()
        y_mat_clean = y_mat.loc[X_transformed_clean.index]

        if 'regime_p_high_vol' in X_transformed_clean.columns:
            prob_vol = X_transformed_clean['regime_p_high_vol']
            X_for_mi = X_transformed_clean.drop(columns=['regime_p_high_vol'])
            _, valid_features = InformationTheory.regime_conditioned_mi(
                X=X_for_mi, y=y_mat_clean, prob_high_vol=prob_vol, mi_thresh=self.mi_threshold
            )
        else:
            mi_scores = mutual_info_classif(X_transformed_clean, y_mat_clean, random_state=42)
            for idx, col in enumerate(X_transformed_clean.columns):
                if mi_scores[idx] > self.mi_threshold:
                    valid_features.append(col)

        # Fallback an toàn nếu MI loại trừ quá mạnh (tránh rỗng ma trận)
        if len(valid_features) < 3:
            print("  -> [Cảnh báo] Bộ lọc MI loại quá gắt, fallback chọn top 5 features.")
            X_for_mi = X_transformed_clean.drop(columns=['regime_p_high_vol'], errors='ignore')
            corr_fallback = X_for_mi.corrwith(y_mat_clean, method='spearman').abs()
            valid_features = corr_fallback.sort_values(ascending=False).head(5).index.tolist()

        return X_transformed[valid_features], optimal_lags, valid_features