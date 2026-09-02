# src/engine/ml_trainer.py

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from .purged_cv import PurgedTimeSeriesSplit

class MLTrainer:
    """Huấn luyện mô hình XGBoost đa lớp với Nested Purged CV[cite: 1, 2]."""
    def __init__(self, n_splits=3, purge_gap=5, random_state=42):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.random_state = random_state

    def train(self, X_train, y_train):
        # Map nhãn (-1, 0, 1) -> (0, 1, 2) cho XGBoost[cite: 2, 3]
        y_train_mapped = y_train + 1

        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Cấu hình không gian siêu tham số
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'reg_alpha': [0.1, 0.5, 1.0, 5.0],
            'reg_lambda': [1.0, 5.0, 10.0]
        }

        inner_cv = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_gap=self.purge_gap)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            scoring='accuracy',
            cv=inner_cv,
            random_state=self.random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train_mapped)
        return random_search.best_estimator_