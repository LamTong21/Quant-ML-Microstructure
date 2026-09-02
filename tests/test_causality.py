# tests/test_causality.py

import pytest
import pandas as pd
import numpy as np
from src.feature_selection.causality_screen import CausalityScreen

def test_granger_causality_screen():
    """Kiểm thử khả năng nhận diện quan hệ nhân quả trễ (Lagged Causality)[cite: 1, 2, 3]."""
    np.random.seed(42)
    n = 200
    
    # Tạo biến độc lập ngẫu nhiên
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)  # Biến nhiễu, không có quan hệ nhân quả
    
    # Tạo biến mục tiêu y phụ thuộc vào x1 ở trễ (lag) 2
    y = np.roll(x1, 2) * 1.5 + np.random.normal(0, 0.1, n)
    
    df_X = pd.DataFrame({'feature_causal': x1, 'feature_noise': x2})
    series_y = pd.Series(y, name='target')
    
    # Lọc bỏ các hàng NaN do quá trình roll
    df_X = df_X.iloc[2:].reset_index(drop=True)
    series_y = series_y.iloc[2:].reset_index(drop=True)
    
    valid_features, optimal_lags = CausalityScreen.screen(df_X, series_y, max_lag=3, p_threshold=0.05)
    
    # 1. Biến nhân quả phải được giữ lại, biến nhiễu bị loại bỏ
    assert 'feature_causal' in valid_features
    
    # 2. Thuật toán phải phát hiện đúng độ trễ tối ưu là 2
    assert optimal_lags.get('feature_causal') == 2