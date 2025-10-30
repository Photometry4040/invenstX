"""
피처 엔지니어링 유틸리티

기술적 지표를 계산하여 강화학습 모델의 State representation을 확장합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """기술적 지표 계산 및 피처 엔지니어링"""

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: OHLCV 데이터프레임
        """
        self.data = data.copy()

    def add_all_features(self,
                        rsi_window: int = 14,
                        macd_fast: int = 12,
                        macd_slow: int = 26,
                        macd_signal: int = 9,
                        bb_window: int = 20,
                        bb_std: int = 2) -> pd.DataFrame:
        """
        모든 필수 피처를 추가합니다.

        Args:
            rsi_window: RSI 계산 윈도우
            macd_fast: MACD 빠른 EMA 기간
            macd_slow: MACD 느린 EMA 기간
            macd_signal: MACD 신호선 기간
            bb_window: Bollinger Bands 윈도우
            bb_std: Bollinger Bands 표준편차 배수

        Returns:
            피처가 추가된 데이터프레임
        """
        df = self.data.copy()

        # 1. RSI
        df['RSI'] = self.calculate_rsi(df['Close'], window=rsi_window)

        # 2. MACD
        macd_data = self.calculate_macd(
            df['Close'],
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal
        )
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['MACD_Signal']
        df['MACD_Histogram'] = macd_data['MACD_Histogram']

        # 3. Bollinger Bands
        bb_data = self.calculate_bollinger_bands(
            df['Close'],
            window=bb_window,
            num_std=bb_std
        )
        df['BB_Upper'] = bb_data['BB_Upper']
        df['BB_Middle'] = bb_data['BB_Middle']
        df['BB_Lower'] = bb_data['BB_Lower']

        # 4. Moving Averages (권장)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # 5. ATR (Average True Range)
        df['ATR'] = self.calculate_atr(df, window=14)

        # 6. Volume Indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # 7. Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # 8. Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

        # 9. Volatility
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

        # 결측치 제거 (초기 계산 기간)
        df = df.dropna()

        return df

    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index) 계산

        Args:
            prices: 가격 시리즈
            window: 계산 윈도우 (기본 14일)

        Returns:
            RSI 값 (0-100)
        """
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series,
                      fast: int = 12,
                      slow: int = 26,
                      signal: int = 9) -> dict:
        """
        MACD (Moving Average Convergence Divergence) 계산

        Args:
            prices: 가격 시리즈
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 신호선 EMA 기간

        Returns:
            MACD, Signal, Histogram을 포함한 딕셔너리
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal

        return {
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Histogram': macd_histogram
        }

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series,
                                  window: int = 20,
                                  num_std: int = 2) -> dict:
        """
        Bollinger Bands 계산

        Args:
            prices: 가격 시리즈
            window: 이동평균 윈도우
            num_std: 표준편차 배수

        Returns:
            Upper, Middle, Lower 밴드를 포함한 딕셔너리
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return {
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower
        }

    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        ATR (Average True Range) 계산

        Args:
            data: OHLC 데이터프레임
            window: 계산 윈도우

        Returns:
            ATR 값
        """
        high = data['High']
        low = data['Low']
        close = data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr


def create_enhanced_dataset(file_path: str,
                           output_path: Optional[str] = None) -> pd.DataFrame:
    """
    CSV 파일에서 데이터를 읽어 피처를 추가하고 저장합니다.

    Args:
        file_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로 (None이면 저장 안함)

    Returns:
        피처가 추가된 데이터프레임
    """
    # 데이터 로드
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # 피처 엔지니어링
    engineer = FeatureEngineer(data)
    enhanced_data = engineer.add_all_features()

    # 저장
    if output_path:
        enhanced_data.to_csv(output_path)
        print(f"Enhanced data saved to {output_path}")
        print(f"Original features: {len(data.columns)}")
        print(f"Enhanced features: {len(enhanced_data.columns)}")
        print(f"Records: {len(data)} → {len(enhanced_data)} (after dropping NaN)")

    return enhanced_data


if __name__ == "__main__":
    # 사용 예시
    print("=" * 80)
    print("Feature Engineering Pipeline")
    print("=" * 80)
    print()

    # cleaned_AAPL.csv에 피처 추가
    enhanced_data = create_enhanced_dataset(
        file_path='data/cleaned_AAPL.csv',
        output_path='data/enhanced_AAPL.csv'
    )

    print()
    print("Added features:")
    print("-" * 80)
    new_features = [col for col in enhanced_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    for i, feat in enumerate(new_features, 1):
        print(f"{i:2d}. {feat}")

    print()
    print("Sample data (last 3 rows):")
    print("-" * 80)
    print(enhanced_data[['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].tail(3))
    print()
    print("✅ Feature engineering complete!")
