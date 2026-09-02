# tests/test_data_integrity.py

import pytest
import pandas as pd
import numpy as np
from src.data_loader.integrity_audit import IntegrityAudit

@pytest.fixture
def mock_raw_data():
    """Tạo dữ liệu OHLCV thô có chứa lỗi hình học, trùng lặp và sự kiện doanh nghiệp."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        'open': [10.0, 10.5, 11.0, 5.5, 6.0],
        'high': [10.2, 10.8, 10.5, 5.8, 6.2],  # Lỗi ở index 2: high < open
        'low': [9.8, 10.0, 10.8, 5.2, 5.8],
        'close': [10.5, 11.0, 10.8, 5.8, 6.1],
        'volume': [1000, 1200, 1500, 2000, 1800],
        'split_factor': [1.0, 1.0, 1.0, 2.0, 1.0],  # Chia tách tỷ lệ 1:2 tại index 3
        'dividend': [0.0, 0.0, 0.0, 0.5, 0.0]       # Cổ tức 0.5 tại index 3
    }, index=dates)
    
    # Tạo bản ghi trùng lặp
    df = pd.concat([df, df.iloc[[-1]]])
    return df

def test_integrity_audit(mock_raw_data):
    """Kiểm thử khử trùng lặp, biên hình học và cơ chế Forward-Adjustment[cite: 1, 2, 3]."""
    df_clean, stats = IntegrityAudit.run_audit(mock_raw_data)
    
    # 1. Kiểm tra khử trùng lặp
    assert len(df_clean) == 5
    assert stats["initial_bars"] == 6
    assert stats["clean_bars"] == 5

    # 2. Kiểm tra biên hình học nến
    assert (df_clean['high'] >= df_clean[['open', 'close']].max(axis=1)).all()
    assert (df_clean['low'] <= df_clean[['open', 'close']].min(axis=1)).all()

    # 3. Kiểm tra Forward-Adjustment
    # Ngày 1 (Gốc): close_adj = close
    assert df_clean['close_adj'].iloc[0] == 10.5
    
    # Ngày 4 (Sự kiện chia tách & cổ tức): 
    # Return thực tế = (5.8 + 0.5) / (10.8 * 2.0) - 1.0 = 6.3 / 21.6 - 1.0 = -0.7083
    # close_adj(t) = close_adj(t-1) * (1 + Return)
    expected_return = (5.8 + 0.5) / (10.8 * 2.0) - 1.0
    expected_adj = df_clean['close_adj'].iloc[2] * (1.0 + expected_return)
    assert np.isclose(df_clean['close_adj'].iloc[3], expected_adj)