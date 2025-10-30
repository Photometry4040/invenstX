"""
쌍봉(Double Top) 패턴 감지 모듈

이 모듈은 주식 차트에서 쌍봉 패턴을 감지하는 기능을 제공합니다.
쌍봉 패턴은 상승 추세에서 발생하는 반전 패턴으로, 두 개의 고점이 비슷한 가격대에서 형성되는 패턴입니다.
"""

import pandas as pd
import numpy as np
from .pattern_analyzer import ChartPattern

class DoubleTop(ChartPattern):
    """
    쌍봉(Double Top) 패턴을 감지하는 클래스
    
    쌍봉 패턴은 상승 추세에서 발생하는 반전 패턴으로, 두 개의 고점이 비슷한 가격대에서 형성되는 패턴입니다.
    이 패턴이 완성되면 하락 추세로의 전환 신호로 해석됩니다.
    """
    
    def __init__(self, data, window=20, threshold=0.03):
        """
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            window (int): 패턴 감지를 위한 기간 (기본값: 20)
            threshold (float): 두 고점 간의 최대 허용 가격 차이 비율 (기본값: 0.03, 즉 3%)
        """
        super().__init__(data)
        self.window = window
        self.threshold = threshold
        self.name = "Double Top"
    
    def detect(self):
        """
        쌍봉 패턴 감지
        
        Returns:
            pd.Series: 패턴 감지 결과 (-1: 하락 신호, 0: 신호 없음)
        """
        # 결과를 저장할 시리즈 초기화
        result = pd.Series(0, index=self.data.index)
        
        # 데이터가 충분하지 않으면 빈 결과 반환
        if len(self.data) < self.window * 2:
            return result
        
        # 데이터 열 추출 및 Series 확인
        high_series = self.data['High']
        low_series = self.data['Low']
        
        if not isinstance(high_series, pd.Series):
            high_series = high_series.iloc[:, 0]
        if not isinstance(low_series, pd.Series):
            low_series = low_series.iloc[:, 0]
        
        try:
            # 고점 찾기
            for i in range(self.window, len(self.data) - self.window):
                # 현재 위치를 중심으로 앞뒤 window 크기의 데이터 확인
                left_high = high_series.iloc[i - self.window:i]
                right_high = high_series.iloc[i:i + self.window]
                
                # 현재 위치가 왼쪽 창에서 최고가인지 확인
                current_high = float(high_series.iloc[i])
                left_max = float(left_high.max())
                
                if current_high >= left_max:
                    # 오른쪽 창에서 비슷한 고점 찾기
                    for j in range(i + 5, i + self.window):  # 최소 5봉 이상 떨어진 위치에서 찾기
                        if j < len(self.data):
                            # 두 번째 고점이 첫 번째 고점과 비슷한지 확인
                            second_high = float(high_series.iloc[j])
                            price_diff = abs(second_high - current_high) / current_high
                            right_max = float(right_high.max())
                            
                            if price_diff <= self.threshold and second_high >= right_max * 0.98:
                                # 두 고점 사이에 저점이 있는지 확인
                                between_low = low_series.iloc[i:j]
                                if len(between_low) > 3:  # 최소 3봉 이상 있어야 함
                                    min_between = float(between_low.min())
                                    min_idx = between_low.idxmin()
                                    
                                    # 저점이 두 고점보다 충분히 낮은지 확인
                                    if (min_between < current_high * 0.97 and 
                                        min_between < second_high * 0.97):
                                        
                                        # 두 번째 고점 이후 가격이 저점 아래로 내려가는지 확인
                                        if j + 3 < len(self.data):
                                            after_low = low_series.iloc[j:j+5]
                                            after_min = float(after_low.min())
                                            if after_min < min_between:
                                                # 하락 신호 설정
                                                result.iloc[j] = -1
        except Exception as e:
            print(f"쌍봉 패턴 감지 중 오류 발생: {e}")
        
        return result
    
    def get_signals(self, style='default'):
        """
        투자 스타일에 따른 매매 신호 생성
        
        Args:
            style (str): 투자 스타일 ('default', 'long_term', 'swing')
            
        Returns:
            pd.DataFrame: 날짜별 매매 신호와 강도
        """
        try:
            # 패턴 감지
            pattern_result = self.detect()
            
            # 신호 및 강도 초기화
            signals = pd.DataFrame(index=pattern_result.index)
            signals['Signal'] = 0
            signals['Strength'] = 0.0
            
            # 패턴이 감지된 위치에 신호 설정
            for i in range(len(pattern_result)):
                if pattern_result.iloc[i] != 0:
                    # 기본 신호 설정
                    signals.iloc[i, signals.columns.get_loc('Signal')] = pattern_result.iloc[i]
                    
                    # 투자 스타일에 따른 신호 강도 조정
                    strength_value = 0.0
                    if style == 'long_term':
                        # 장기 투자: 강도 감소
                        strength_value = abs(float(pattern_result.iloc[i])) * 0.7
                    elif style == 'swing':
                        # 스윙 트레이드: 강도 증가
                        strength_value = abs(float(pattern_result.iloc[i])) * 1.2
                    else:
                        # 기본 강도
                        strength_value = abs(float(pattern_result.iloc[i]))
                    
                    signals.iloc[i, signals.columns.get_loc('Strength')] = strength_value
            
            return signals
            
        except Exception as e:
            # 오류 발생 시 빈 신호 데이터프레임 반환
            print(f"쌍봉 패턴 신호 생성 중 오류 발생: {e}")
            empty_signals = pd.DataFrame(index=self.data.index)
            empty_signals['Signal'] = 0
            empty_signals['Strength'] = 0.0
            return empty_signals 