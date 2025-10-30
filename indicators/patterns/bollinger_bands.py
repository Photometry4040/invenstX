"""
볼린저 밴드 패턴 인식기

볼린저 밴드를 활용한 패턴 인식 및 매매 신호 생성 클래스입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .pattern_analyzer import Pattern

class BollingerBands(Pattern):
    """
    볼린저 밴드 패턴 인식 클래스
    
    볼린저 밴드를 계산하고 이를 기반으로 매매 신호를 생성합니다.
    """
    
    def __init__(self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0):
        """
        볼린저 밴드 패턴 인식기 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            window (int, optional): 이동 평균 기간. 기본값은 20.
            num_std (float, optional): 표준 편차 배수. 기본값은 2.0.
        """
        super().__init__(data)
        self.window = window
        self.num_std = num_std
        self.name = "BollingerBands"
        self._calculate_bands()
        
    def _calculate_bands(self) -> None:
        """
        볼린저 밴드 계산
        """
        # 데이터 타입 확인 및 변환
        close_series = self.data['Close'] if isinstance(self.data['Close'], pd.Series) else self.data['Close'].iloc[:, 0]
        
        # 이동 평균 계산
        self.data['MA'] = close_series.rolling(window=self.window).mean()
        
        # 표준 편차 계산
        self.data['STD'] = close_series.rolling(window=self.window).std()
        
        # 밴드 계산
        self.data['UpperBand'] = self.data['MA'] + (self.data['STD'] * self.num_std)
        self.data['LowerBand'] = self.data['MA'] - (self.data['STD'] * self.num_std)
        
        # 밴드 폭 계산
        self.data['BandWidth'] = (self.data['UpperBand'] - self.data['LowerBand']) / self.data['MA']
        
        # 볼린저 밴드 %B 계산
        # %B = (가격 - 하단 밴드) / (상단 밴드 - 하단 밴드)
        upper_band = self.data['UpperBand'] if isinstance(self.data['UpperBand'], pd.Series) else self.data['UpperBand'].iloc[:, 0]
        lower_band = self.data['LowerBand'] if isinstance(self.data['LowerBand'], pd.Series) else self.data['LowerBand'].iloc[:, 0]
        
        self.data['PercentB'] = (close_series - lower_band) / (upper_band - lower_band)
        
    def detect(self) -> pd.DataFrame:
        """
        볼린저 밴드 패턴 감지
        
        Returns:
            pd.DataFrame: 패턴이 감지된 인덱스와 신호 강도가 포함된 데이터프레임
        """
        # 결과 데이터프레임 초기화
        result = pd.DataFrame(index=self.data.index)
        result['Pattern'] = None
        result['Strength'] = 0.0
        
        # Series 변환 확인
        percent_b = self.data['PercentB'] if isinstance(self.data['PercentB'], pd.Series) else self.data['PercentB'].iloc[:, 0]
        band_width = self.data['BandWidth'] if isinstance(self.data['BandWidth'], pd.Series) else self.data['BandWidth'].iloc[:, 0]
        close = self.data['Close'] if isinstance(self.data['Close'], pd.Series) else self.data['Close'].iloc[:, 0]
        upper_band = self.data['UpperBand'] if isinstance(self.data['UpperBand'], pd.Series) else self.data['UpperBand'].iloc[:, 0]
        lower_band = self.data['LowerBand'] if isinstance(self.data['LowerBand'], pd.Series) else self.data['LowerBand'].iloc[:, 0]
        
        # 1. 과매수/과매도 패턴
        # %B가 0보다 작으면 과매도 (하단 밴드 아래)
        # %B가 1보다 크면 과매수 (상단 밴드 위)
        oversold_mask = percent_b < 0
        result.loc[oversold_mask, 'Pattern'] = 'Oversold'
        if any(oversold_mask):
            result.loc[oversold_mask, 'Strength'] = 1 - percent_b.loc[oversold_mask]
        
        overbought_mask = percent_b > 1
        result.loc[overbought_mask, 'Pattern'] = 'Overbought'
        if any(overbought_mask):
            result.loc[overbought_mask, 'Strength'] = percent_b.loc[overbought_mask] - 1
        
        # 2. 밴드 폭 수축 패턴 (변동성 축소)
        # 밴드 폭이 20일 중 최소값에 가까우면 변동성 축소로 간주
        band_width_min = band_width.rolling(window=20).min()
        band_width_max = band_width.rolling(window=20).max()
        band_width_range = band_width_max - band_width_min
        
        # 밴드 폭이 20일 최소값의 10% 이내인 경우
        is_band_width_low = (band_width - band_width_min) / band_width_range < 0.1
        result.loc[is_band_width_low, 'Pattern'] = 'BandwidthContraction'
        if any(is_band_width_low):
            result.loc[is_band_width_low, 'Strength'] = 1 - (band_width.loc[is_band_width_low] - band_width_min.loc[is_band_width_low]) / band_width_range.loc[is_band_width_low]
        
        # 3. 밴드 반전 패턴
        # 하단 밴드 이탈 후 다시 상승하는 패턴
        lower_band_cross = (close.shift(1) < lower_band.shift(1)) & (close > lower_band)
        result.loc[lower_band_cross, 'Pattern'] = 'LowerBandReversal'
        result.loc[lower_band_cross, 'Strength'] = 1.0
        
        # 상단 밴드 이탈 후 다시 하락하는 패턴
        upper_band_cross = (close.shift(1) > upper_band.shift(1)) & (close < upper_band)
        result.loc[upper_band_cross, 'Pattern'] = 'UpperBandReversal'
        result.loc[upper_band_cross, 'Strength'] = 1.0
        
        return result
    
    def get_signals(self) -> pd.DataFrame:
        """
        볼린저 밴드 기반 매매 신호 생성
        
        Returns:
            pd.DataFrame: 매매 신호가 포함된 데이터프레임
                - 1: 매수 신호
                - 0: 홀드 신호
                - -1: 매도 신호
        """
        # 패턴 감지
        patterns = self.detect()
        
        # 결과 데이터프레임 초기화
        signals = pd.DataFrame(index=self.data.index)
        signals['Signal'] = 0
        
        # Series 변환 확인
        close = self.data['Close'] if isinstance(self.data['Close'], pd.Series) else self.data['Close'].iloc[:, 0]
        upper_band = self.data['UpperBand'] if isinstance(self.data['UpperBand'], pd.Series) else self.data['UpperBand'].iloc[:, 0]
        volume = self.data['Volume'] if isinstance(self.data['Volume'], pd.Series) else self.data['Volume'].iloc[:, 0]
        percent_b = self.data['PercentB'] if isinstance(self.data['PercentB'], pd.Series) else self.data['PercentB'].iloc[:, 0]
        
        # 1. 과매도 구간에서 매수 신호
        oversold = patterns['Pattern'] == 'Oversold'
        signals.loc[oversold, 'Signal'] = 1
        
        # 매수 신호 강도 조정 (과매도 강도에 비례)
        if any(oversold):
            signals.loc[oversold, 'Signal'] = patterns.loc[oversold, 'Strength']
        
        # 2. 과매수 구간에서 매도 신호
        overbought = patterns['Pattern'] == 'Overbought'
        signals.loc[overbought, 'Signal'] = -1
        
        # 매도 신호 강도 조정 (과매수 강도에 비례)
        if any(overbought):
            signals.loc[overbought, 'Signal'] = -patterns.loc[overbought, 'Strength']
        
        # 3. 밴드 폭 수축 후 돌파 신호
        band_contraction = patterns['Pattern'] == 'BandwidthContraction'
        
        # 밴드 폭 수축 후 거래량 증가와 함께 상승 돌파하면 매수 신호
        for i in range(len(signals) - 1):
            if band_contraction.iloc[i]:
                # 다음날 상단 밴드 돌파 및 거래량 증가 확인
                if i + 1 < len(signals) and close.iloc[i + 1] > upper_band.iloc[i + 1] and \
                   volume.iloc[i + 1] > volume.iloc[i] * 1.5:
                    signals.loc[signals.index[i + 1], 'Signal'] = 1
        
        # 4. 하단 밴드 반전 패턴에서 매수 신호
        lower_reversal = patterns['Pattern'] == 'LowerBandReversal'
        signals.loc[lower_reversal, 'Signal'] = 0.8  # 매수 신호 (강도 0.8)
        
        # 5. 상단 밴드 반전 패턴에서 매도 신호
        upper_reversal = patterns['Pattern'] == 'UpperBandReversal'
        signals.loc[upper_reversal, 'Signal'] = -0.8  # 매도 신호 (강도 -0.8)
        
        # 볼린저 밴드 %B 기반 신호 보강
        for i in range(len(signals)):
            # %B가 0.05 미만이면 매수 신호 (강한 과매도)
            if 0 < percent_b.iloc[i] < 0.05:
                signals.loc[signals.index[i], 'Signal'] = max(signals.loc[signals.index[i], 'Signal'], 0.9)
            
            # %B가 0.95 초과이면 매도 신호 (강한 과매수)
            elif 0.95 < percent_b.iloc[i] < 1:
                signals.loc[signals.index[i], 'Signal'] = min(signals.loc[signals.index[i], 'Signal'], -0.9)
        
        return signals 