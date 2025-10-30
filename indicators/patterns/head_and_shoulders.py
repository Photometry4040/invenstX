"""
헤드앤숄더(Head and Shoulders) 패턴 감지 모듈

이 모듈은 주식 차트에서 헤드앤숄더 패턴을 감지하는 기능을 제공합니다.
헤드앤숄더 패턴은 상승 추세에서 발생하는 반전 패턴으로, 세 개의 고점이 형성되며 가운데 고점이 가장 높은 패턴입니다.
"""

import pandas as pd
import numpy as np
from .pattern_analyzer import ChartPattern

class HeadAndShoulders(ChartPattern):
    """
    헤드앤숄더(Head and Shoulders) 패턴을 감지하는 클래스
    
    헤드앤숄더 패턴은 상승 추세에서 발생하는 반전 패턴으로, 세 개의 고점이 형성되며 가운데 고점이 가장 높은 패턴입니다.
    이 패턴이 완성되면 하락 추세로의 전환 신호로 해석됩니다.
    """
    
    def __init__(self, data, window=20, threshold=0.03):
        """
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            window (int): 패턴 감지를 위한 기간 (기본값: 20)
            threshold (float): 좌우 어깨 간의 최대 허용 가격 차이 비율 (기본값: 0.03, 즉 3%)
        """
        super().__init__(data)
        self.window = window
        self.threshold = threshold
        self.name = "Head and Shoulders"
    
    def detect(self):
        """
        헤드앤숄더 패턴 감지
        
        Returns:
            pd.Series: 패턴 감지 결과 (-1: 하락 신호, 0: 신호 없음)
        """
        # 결과를 저장할 시리즈 초기화
        result = pd.Series(0, index=self.data.index)
        
        # 데이터가 충분하지 않으면 빈 결과 반환
        if len(self.data) < self.window * 3:
            return result
        
        # 데이터 열 추출 및 Series 확인
        high_series = self.data['High']
        low_series = self.data['Low']
        close_series = self.data['Close']
        
        if not isinstance(high_series, pd.Series):
            high_series = high_series.iloc[:, 0]
        if not isinstance(low_series, pd.Series):
            low_series = low_series.iloc[:, 0]
        if not isinstance(close_series, pd.Series):
            close_series = close_series.iloc[:, 0]
        
        # 고점 찾기
        for i in range(self.window, len(self.data) - self.window * 2):
            # 왼쪽 어깨 후보
            left_shoulder_idx = i
            left_shoulder_price = float(high_series.iloc[i])
            
            # 왼쪽 어깨가 주변에서 고점인지 확인
            left_window = high_series.iloc[i - self.window:i + self.window]
            left_window_max = float(left_window.max())
            if left_shoulder_price < left_window_max:
                continue
            
            # 헤드 후보 찾기
            for j in range(i + 5, i + self.window):
                if j >= len(self.data) - self.window:
                    break
                    
                head_idx = j
                head_price = float(high_series.iloc[j])
                
                # 헤드가 왼쪽 어깨보다 높은지 확인
                if head_price <= left_shoulder_price:
                    continue
                
                # 헤드가 주변에서 고점인지 확인
                head_window = high_series.iloc[j - 5:j + 5]
                head_window_max = float(head_window.max())
                if head_price < head_window_max:
                    continue
                
                # 오른쪽 어깨 후보 찾기
                for k in range(j + 5, j + self.window):
                    if k >= len(self.data):
                        break
                        
                    right_shoulder_idx = k
                    right_shoulder_price = float(high_series.iloc[k])
                    
                    # 오른쪽 어깨가 헤드보다 낮은지 확인
                    if right_shoulder_price >= head_price:
                        continue
                    
                    # 오른쪽 어깨가 왼쪽 어깨와 비슷한지 확인
                    price_diff = abs(right_shoulder_price - left_shoulder_price) / left_shoulder_price
                    if price_diff > self.threshold:
                        continue
                    
                    # 오른쪽 어깨가 주변에서 고점인지 확인
                    right_window = high_series.iloc[k - 5:k + 5] if k + 5 < len(self.data) else high_series.iloc[k - 5:]
                    right_window_max = float(right_window.max())
                    if k + 5 < len(self.data) and right_shoulder_price < right_window_max:
                        continue
                    
                    try:
                        # 목선(neckline) 확인
                        left_section = low_series.iloc[i:j]
                        right_section = low_series.iloc[j:k]
                        
                        left_low_idx = left_section.idxmin()
                        right_low_idx = right_section.idxmin()
                        
                        left_low = float(self.data.loc[left_low_idx, 'Low'])
                        right_low = float(self.data.loc[right_low_idx, 'Low'])
                        
                        # 목선 기울기 계산
                        left_low_pos = self.data.index.get_loc(left_low_idx)
                        right_low_pos = self.data.index.get_loc(right_low_idx)
                        neckline_slope = (right_low - left_low) / (right_low_pos - left_low_pos)
                        
                        # 목선 돌파 확인
                        if k + 3 < len(self.data):
                            # 오른쪽 어깨 이후 데이터
                            for m in range(5):
                                if k + m >= len(self.data):
                                    break
                                    
                                idx = k + m
                                days_from_right_low = idx - right_low_pos
                                neckline_value = right_low + neckline_slope * days_from_right_low
                                
                                # 가격이 목선 아래로 내려가는지 확인
                                current_close = float(close_series.iloc[idx])
                                if current_close < neckline_value:
                                    # 하락 신호 설정
                                    result.iloc[idx] = -1
                                    break
                    except Exception as e:
                        # 오류 발생 시 계속 진행
                        print(f"헤드앤숄더 패턴 감지 중 오류 발생: {e}")
                        continue
        
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
                        # 장기 투자: 강도 증가 (헤드앤숄더는 강한 반전 신호)
                        strength_value = abs(float(pattern_result.iloc[i])) * 1.3
                    elif style == 'swing':
                        # 스윙 트레이드: 강도 증가
                        strength_value = abs(float(pattern_result.iloc[i])) * 1.5
                    else:
                        # 기본 강도
                        strength_value = abs(float(pattern_result.iloc[i])) * 1.2
                    
                    signals.iloc[i, signals.columns.get_loc('Strength')] = strength_value
            
            return signals
            
        except Exception as e:
            # 오류 발생 시 빈 신호 데이터프레임 반환
            print(f"헤드앤숄더 패턴 신호 생성 중 오류 발생: {e}")
            empty_signals = pd.DataFrame(index=self.data.index)
            empty_signals['Signal'] = 0
            empty_signals['Strength'] = 0.0
            return empty_signals 