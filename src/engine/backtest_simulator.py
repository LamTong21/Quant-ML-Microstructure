# src/engine/backtest_simulator.py

import numpy as np
import pandas as pd
from scipy import stats

class VectorizedBacktestSimulator:
    """
    Giả lập Backtest có tính đến độ trễ khớp lệnh (Execution Lag) và chi phí giao dịch.
    Chuyển đổi tín hiệu thành Position sizing và tính Net P&L[cite: 1, 2, 3].
    """
    def __init__(self, fee_bps: float = 15.0, slippage_bps: float = 5.0, execution_lag: int = 1):
        # Tổng chi phí 1 chiều giao dịch danh nghĩa (Ví dụ: 20 bps = 0.2%)[cite: 1, 2, 3]
        self.tc = (fee_bps + slippage_bps) / 10000.0
        self.lag = execution_lag

    def run(self, predictions: pd.Series, df_market: pd.DataFrame) -> dict:
        # Căn chỉnh dữ liệu
        aligned = pd.concat([predictions, df_market['log_return']], axis=1, join='inner').dropna()
        y_pred = aligned.iloc[:, 0]
        daily_return = aligned.iloc[:, 1] # Lợi nhuận sinh ra trong ngày t

        # 1. EXECUTION LAG
        # Tín hiệu có ở cuối ngày t, vị thế thực tế chỉ bắt đầu có tác dụng từ ngày t+1[cite: 1, 2, 3]
        actual_position = y_pred.shift(self.lag).fillna(0)

        # 2. TURNOVER & TRANSACTION COSTS
        # Tính sự thay đổi vị thế (Turnover = |W_t - W_{t-1}|)[cite: 1, 2, 3]
        position_change = actual_position.diff().fillna(0)
        turnover = position_change.abs()
        transaction_costs = turnover * self.tc

        # 3. MARK-TO-MARKET P&L
        # Lợi nhuận gộp = Vị thế giữ qua đêm x Lợi nhuận ngày hôm sau
        gross_strategy_return = actual_position * daily_return

        # Lợi nhuận ròng sau phí (Net Strategy Return)[cite: 1, 2]
        net_strategy_return = gross_strategy_return - transaction_costs
        equity_curve = np.exp(net_strategy_return.cumsum())

        # 4. RISK METRICS (Annualized)[cite: 1, 2, 3]
        mean_ret = net_strategy_return.mean() * 252
        std_ret = net_strategy_return.std() * np.sqrt(252)
        net_sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # Maximum Drawdown (MDD)
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        mdd = drawdown.min()

        # 5. Rank IC (Đánh giá sức mạnh dự báo nguyên thủy)[cite: 1, 2, 3]
        actual_h_day_ret = daily_return.rolling(5).sum().shift(-5).dropna()
        valid_idx = y_pred.index.intersection(actual_h_day_ret.index)
        if len(valid_idx) > 0:
            ic, _ = stats.spearmanr(y_pred.loc[valid_idx], actual_h_day_ret.loc[valid_idx])
        else:
            ic = np.nan

        return {
            'N_Trades': turnover.sum() / 2, # Chia 2 vì 1 round-trip gồm vào và ra
            'Rank_IC': ic,
            'Net_Annualized_Return': mean_ret,
            'Net_Sharpe': net_sharpe,
            'Max_Drawdown': mdd,
            'Calmar_Ratio': mean_ret / abs(mdd) if mdd < 0 else np.inf,
            'Total_Turnover': turnover.sum(),
            'Equity_Curve': equity_curve
        }