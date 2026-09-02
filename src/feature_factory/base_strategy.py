# src/feature_factory/base_strategy.py

import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatureStrategy(ABC):
    """
    Abstract Base Class cho tất cả các chiến lược thiết kế đặc trưng.
    Đảm bảo tính nhân quả kinh tế lượng và không gây rò rỉ dữ liệu.
    """
    @abstractmethod
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        """
        Thực thi tính toán ma trận đặc trưng.
        
        Args:
            df (pd.DataFrame): Dataframe chứa dữ liệu OHLCV và các biến đã tiền xử lý.
            eps (float): Epsilon nhỏ để tránh lỗi chia cho 0.
            
        Returns:
            pd.DataFrame: Dataframe chứa các đặc trưng mới được tạo.
        """
        pass