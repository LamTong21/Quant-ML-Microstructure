# tests/test_backtest_engine.py

import pytest
import pandas as pd
import numpy as np
from src.engine.backtest_simulator import VectorizedBacktestSimulator

@pytest.fixture
def mock_prediction_data():
    """Tạo chuỗi tín hiệu và lợi nhuận thị trường giả lập."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    
    # Tín hiệu vị thế: Hold Cash, Long, Long, Short, Short, Hold Cash
    predictions = pd.Series([0, 1, 1, -1, -1, 0], index=dates, name='predicted_target')
    
    # Market return
    market_df = pd.DataFrame({
        'log_return': [0.01, 0.02, -0.01, -0.02, 0.03, -0.01]
    }, index=dates)
    
    return predictions, market_df

def test_backtest_simulator_execution_and_frictions(mock_prediction_data):
    """Kiểm thử độ trễ thực thi t+1 và khấu trừ phí ma sát giao dịch[cite: 1, 2, 3]."""
    preds, df_market = mock_prediction_data
    
    # Cấu hình phí 20 bps (15 bps môi giới + 5 bps trượt giá) và độ trễ t+1[cite: 1, 2, 3]
    simulator = VectorizedBacktestSimulator(fee_bps=15.0, slippage_bps=5.0, execution_lag=1)
    results = simulator.run(preds, df_market)
    
    # 1. Kiểm tra số lượng giao dịch (N_Trades)
    # Tín hiệu thực tế (shifted 1): [NaN, 0, 1, 1, -1, -1]
    # Diff (Turnover): [NaN, 0, 1, 0, 2, 0] => Tổng Turnover = 3
    # N_Trades = Turnover / 2 = 1.5 round-trips
    assert results['Total_Turnover'] == 3.0
    assert results['N_Trades'] == 1.5
    
    # 2. Kiểm tra P&L Mark-to-Market tại ngày 3
    # Ngày 3 (Index 2): actual_position = 1, market_return = -0.01.
    # position_change = |1 - 0| = 1. transaction_cost = 1 * 0.0020 = 0.0020.
    # strat_return = (1 * -0.01) - 0.0020 = -0.0120.
    equity_curve = results['Equity_Curve']
    expected_return_day_3 = -0.0120
    
    # Equity curve là tích lũy exp(cumsum)
    # Do các ngày trước vị thế = 0, equity curve tại ngày 3 phải là exp(-0.0120)
    assert np.isclose(equity_curve.iloc[2], np.exp(expected_return_day_3))