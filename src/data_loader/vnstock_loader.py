# src/data_loader/vnstock_loader.py

import pandas as pd
import numpy as np
from vnstock import Quote

def extract_daily_microstructure(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp các chỉ số vi cấu trúc từ dữ liệu nội phiên (Intraday)
    thành các biến EOD Epsilon-Causal phục vụ Model.
    """
    if df_intraday.empty:
        return pd.DataFrame()

    df = df_intraday.copy()
    df['date'] = pd.to_datetime(df['time']).dt.date

    # 1. Giả lập / Ước tính Order Flow Imbalance (OFI) từ nến nội phiên
    price_diff = df['close'].diff().fillna(0)
    buy_vol = np.where(price_diff >= 0, df['volume'], 0)
    sell_vol = np.where(price_diff < 0, df['volume'], 0)
    df['order_imbalance'] = buy_vol - sell_vol

    # 2. Tổng hợp theo từng ngày
    daily_agg = df.groupby('date').agg(
        intraday_ofi_mean=('order_imbalance', 'mean'),
        intraday_ofi_std=('order_imbalance', 'std'),
        intraday_ofi_sum=('order_imbalance', 'sum'),
        intraday_vol_sum=('volume', 'sum'),
        intraday_tick_count=('volume', 'count'),
        intraday_high=('high', 'max'),
        intraday_low=('low', 'min')
    ).reset_index()

    # VPIN Proxy (Xác suất độc hại dòng lệnh)
    daily_agg['daily_vpin'] = daily_agg['intraday_ofi_sum'].abs() / (daily_agg['intraday_vol_sum'] + 1e-8)
    daily_agg['intraday_ofi_std'] = daily_agg['intraday_ofi_std'].fillna(0)

    daily_agg.rename(columns={'date': 'time'}, inplace=True)
    daily_agg['time'] = pd.to_datetime(daily_agg['time'])
    return daily_agg

class VnstockLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str, source: str = 'KBS'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.source = source

    def fetch_data(self) -> pd.DataFrame:
        print(f"[*] Bắt đầu tải dữ liệu cho mã {self.symbol}...")
        quote = Quote(symbol=self.symbol, source=self.source)

        print("  -> Đang tải dữ liệu Daily (1D)...")
        df_daily = quote.history(
            start=self.start_date,
            end=self.end_date,
            interval='1D'
        )

        if df_daily is None or df_daily.empty:
            raise ValueError(f"Không có dữ liệu Daily cho mã {self.symbol}")

        df_daily['time'] = pd.to_datetime(df_daily['time'])

        print("  -> Đang tải dữ liệu Intraday (15m) để trích xuất vi cấu trúc...")
        try:
            df_intra = quote.history(
                start=self.start_date,
                end=self.end_date,
                interval='15m'
            )

            if not df_intra.empty:
                df_micro = extract_daily_microstructure(df_intra)
                df_daily = pd.merge(df_daily, df_micro, on='time', how='left')
                print("  ✓ Đã hợp nhất thành công dữ liệu Microstructure.")
            else:
                print("  [!] Dữ liệu Intraday rỗng, chỉ dùng OHLCV cơ bản.")
        except Exception as e_intra:
            print(f"  [!] Không lấy được dữ liệu vi cấu trúc: {e_intra}")

        df_daily.sort_values(by='time', inplace=True)
        
        # Điền khuyết (Forward Fill) cho các biến vi cấu trúc
        micro_cols = [c for c in df_daily.columns if 'intraday' in c or 'vpin' in c]
        if micro_cols:
            df_daily[micro_cols] = df_daily[micro_cols].ffill().fillna(0)

        return df_daily