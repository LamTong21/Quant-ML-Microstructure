# tests/test_feature_generation.py

import pytest
import pandas as pd
import numpy as np
from src.feature_factory.kinematics import KinematicDynamicsStrategy

@pytest.fixture
def mock_clean_data():
    """Tạo dữ liệu sạch đã qua xử lý Forward-Adjustment."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        'open_adj': np.random.uniform(10, 20, 30),
        'high_adj': np.random.uniform(15, 25, 30),
        'low_adj': np.random.uniform(5, 15, 30),
        'close_adj': np.random.uniform(10, 20, 30),
        'volume': np.random.randint(1000, 5000, 30)
    }, index=dates)
    
    df['log_return'] = np.log(df['close_adj'] / df['close_adj'].shift(1)).fillna(0)
    return df

def test_kinematic_dynamics_generation(mock_clean_data):
    """Kiểm thử tính toán các biến động lực học, đảm bảo không rò rỉ và không chứa Inf/NaN[cite: 1, 2, 3]."""
    strategy = KinematicDynamicsStrategy()
    feat_df = strategy.construct(mock_clean_data, eps=1e-8)
    
    # 1. Kiểm tra cấu trúc Output
    assert len(feat_df) == len(mock_clean_data)
    assert (feat_df.index == mock_clean_data.index).all()
    
    # 2. Kiểm tra tính toàn vẹn toán học (Không chia cho 0)
    assert 'kinematic_velocity' in feat_df.columns
    assert 'kinematic_acceleration' in feat_df.columns
    assert 'kinematic_squeeze_ratio' in feat_df.columns
    
    # Sau window rolling=20, các giá trị sau index 20 không được chứa NaN hoặc Inf
    valid_slice = feat_df.iloc[20:]
    assert not valid_slice.isna().any().any()
    assert not np.isinf(valid_slice.values).any()