# src/engine/purged_cv.py

from sklearn.model_selection import TimeSeriesSplit

class PurgedTimeSeriesSplit:
    """
    Kỹ thuật chia Walk-Forward có tích hợp Purging để chặn rò rỉ
    thông tin tương lai do nhãn dự báo (Target) chồng lấn[cite: 2, 3].
    """
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None, groups=None):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        base_splits = list(tscv.split(X))

        for train_idx, test_idx in base_splits:
            # Purge: Bỏ qua h điểm cuối của Train Set tránh hiện tượng chồng lấn nhãn[cite: 2, 3]
            train_idx_purged = train_idx[train_idx < (test_idx[0] - self.purge_gap)]
            if len(train_idx_purged) > 0:
                yield train_idx_purged, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Hàm bắt buộc phải có để Scikit-learn nhận diện Cross Validator."""
        return self.n_splits