# src/engine/signal_generator.py

import numpy as np
import pandas as pd

class SignalGenerator:
    """Lọc xác suất tin cậy cao, Đảo chiều tín hiệu & Lọc rủi ro chế độ cứng[cite: 1, 2]."""
    def __init__(self, confidence_threshold=0.55):
        self.confidence_threshold = confidence_threshold

    def generate(self, model, X_test):
        probs = model.predict_proba(X_test)
        
        prob_short = probs[:, 0]
        prob_neutral = probs[:, 1]
        prob_long = probs[:, 2]

        # 1. Ngưỡng tin cậy xác suất (Probability Thresholding)[cite: 1, 2]
        y_pred_raw = np.zeros(len(X_test))
        y_pred_raw[prob_long > self.confidence_threshold] = 1
        y_pred_raw[prob_short > self.confidence_threshold] = -1

        # 2. Đảo nghịch tín hiệu (Contrarian Flip)
        y_pred_traded = -1 * y_pred_raw

        # 3. Adaptive GMM Regime Hard Filter[cite: 1, 2]
        if 'regime_p_high_vol' in X_test.columns:
            regime_probs = X_test['regime_p_high_vol'].values
            # Nếu xác suất pha rủi ro > 0.50 => Đóng băng giao dịch (Position = 0)[cite: 1, 2]
            y_pred_filtered = np.where(regime_probs > 0.50, 0, y_pred_traded)
            test_regime_prob = X_test['regime_p_high_vol']
        else:
            y_pred_filtered = y_pred_traded
            test_regime_prob = pd.Series(0, index=X_test.index)

        return pd.DataFrame({
            'predicted_target': y_pred_filtered,
            'model_raw_target': y_pred_raw,
            'prob_long': prob_long,
            'prob_short': prob_short,
            'regime_p_high_vol': test_regime_prob
        }, index=X_test.index)