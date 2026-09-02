# src/feature_selection/redundancy_control.py

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class RedundancyControl:
    @staticmethod
    def linear_vif_prune(X: pd.DataFrame, threshold: float = 0.85, max_vif: float = 5.0) -> pd.DataFrame:
        """Nhánh Tuyến tính: Khử tương quan cặp và Đa cộng tuyến (VIF)."""
        X_curr = X.copy()

        # 1. Khử tương quan cặp (Pearson)
        corr_matrix = X_curr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(upper[col] > threshold)]
        X_curr = X_curr.drop(columns=to_drop_corr)

        # 2. Khử VIF > 5.0
        while True:
            if X_curr.shape[1] <= 1:
                break
            vifs = []
            cols = X_curr.columns
            X_vals = X_curr.values
            for j in range(len(cols)):
                y_target = X_vals[:, j]
                X_other = np.delete(X_vals, j, axis=1)
                X_mat = np.column_stack([np.ones(len(X_other)), X_other])

                try:
                    beta, _, _, _ = np.linalg.lstsq(X_mat, y_target, rcond=None)
                    preds = X_mat @ beta
                    ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
                    ss_res = np.sum((y_target - preds) ** 2)
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
                    vif = 1.0 / (1.0 - np.clip(r2, 0, 0.9999))
                except np.linalg.LinAlgError:
                    vif = max_vif + 1.0
                vifs.append(vif)

            max_vif_val = max(vifs)
            if max_vif_val > max_vif:
                drop_idx = np.argmax(vifs)
                X_curr = X_curr.drop(columns=[cols[drop_idx]])
            else:
                break
        return X_curr

    @staticmethod
    def nonlinear_hrp_prune(X: pd.DataFrame, max_clusters: int = 15) -> pd.DataFrame:
        """Nhánh Phi tuyến: Phân cụm phân cấp HRP với liên kết Ward và chọn Medoids."""
        if X.shape[1] <= max_clusters:
            return X

        # Sử dụng Spearman để đo lường quan hệ đơn điệu phi tuyến
        corr = X.corr(method='spearman').clip(-1.0, 1.0).fillna(0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))

        dist_sym = (dist + dist.T) / 2.0
        np.fill_diagonal(dist_sym.values, 0)

        condensed_dist = squareform(dist_sym, checks=False)
        link = linkage(condensed_dist, method='ward')
        clusters = fcluster(link, t=max_clusters, criterion='maxclust')

        selected_medoids = []
        for c_id in np.unique(clusters):
            cluster_cols = X.columns[clusters == c_id]
            if len(cluster_cols) == 1:
                selected_medoids.append(cluster_cols[0])
            else:
                sub_dist = dist_sym.loc[cluster_cols, cluster_cols]
                # Chọn feature nằm ở trung tâm cụm (medoid)
                medoid = sub_dist.sum(axis=1).idxmin()
                selected_medoids.append(medoid)

        return X[selected_medoids]