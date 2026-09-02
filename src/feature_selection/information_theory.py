# src/feature_selection/information_theory.py

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

class InformationTheory:
    @staticmethod
    def regime_conditioned_mi(X: pd.DataFrame, y: pd.Series, prob_high_vol: pd.Series, mi_thresh: float = 0.01) -> tuple:
        """Sàng lọc sức mạnh thông tin dự báo phân rã theo pha thị trường[cite: 1, 2, 3]."""
        mask_high_vol = prob_high_vol > 0.5
        mask_low_vol = prob_high_vol <= 0.5

        mi_high = mutual_info_classif(X[mask_high_vol], y[mask_high_vol], random_state=42)
        mi_low = mutual_info_classif(X[mask_low_vol], y[mask_low_vol], random_state=42)
        mi_all = mutual_info_classif(X, y, random_state=42)

        valid_features = []
        results = []

        for idx, col in enumerate(X.columns):
            is_accepted = (mi_high[idx] > mi_thresh) or (mi_low[idx] > mi_thresh)
            if is_accepted:
                valid_features.append(col)

            results.append({
                'feature': col,
                'mi_high_vol': mi_high[idx],
                'mi_low_vol': mi_low[idx],
                'mi_all': mi_all[idx],
                'accepted': is_accepted
            })

        return pd.DataFrame(results).set_index('feature'), valid_features