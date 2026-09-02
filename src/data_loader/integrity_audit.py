# src/data_loader/integrity_audit.py

import pandas as pd
import numpy as np

class IntegrityAudit:
    @staticmethod
    def run_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        initial_len = len(df)

        # 1. Deduplication & Chronological Sort
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        # 2. Geometric bounds audit (Kiểm toán biên hình học)
        max_oc = df[['open', 'close']].max(axis=1)
        min_oc = df[['open', 'close']].min(axis=1)
        df['high'] = np.maximum(df['high'], max_oc)
        df['low'] = np.minimum(df['low'], min_oc)
        df = df[(df['low'] > 0) & (df['volume'] >= 0)]

        # 3. Forward-Adjustment (Xử lý corporate actions tịnh tiến)
        if 'split_factor' not in df.columns:
            df['split_factor'] = 1.0
        if 'dividend' not in df.columns:
            df['dividend'] = 0.0

        daily_ret = ((df['close'] + df['dividend']) / (df['close'].shift(1) * df['split_factor'])) - 1.0
        
        df['close_adj'] = df['close'].iloc[0] * (1.0 + daily_ret).cumprod()
        df['close_adj'] = df['close_adj'].fillna(df['close'])

        ratio = df['close_adj'] / df['close']
        for col in ['open', 'high', 'low']:
            df[f'{col}_adj'] = df[col] * ratio

        return df, {"initial_bars": initial_len, "clean_bars": len(df)}