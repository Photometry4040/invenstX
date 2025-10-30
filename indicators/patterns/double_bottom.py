"""
쌍바닥(Double Bottom) 패턴 인식기

W자 형태의 쌍바닥 패턴을 감지하고 이를 기반으로 매매 신호를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .pattern_analyzer import Pattern

class DoubleBottom(Pattern):
    """
    쌍바닥(Double Bottom) 패턴 인식 클래스
    
    쌍바닥 패턴은 주가가 두 번 유사한 수준에서 지지력을 보이는 W자 형태의 패턴으로,
    주로 상승 반전 신호로 해석됩니다.
    """
    
    def __init__(self, data: pd.DataFrame, window: int = 20, tolerance: float = 0.03):
        """
        쌍바닥 패턴 인식기 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            window (int, optional): 국소 저점을 찾기 위한 기간. 기본값은 20.
            tolerance (float, optional): 두 저점 간 허용 가능한 가격 차이(%). 기본값은 3%.
        """
        super().__init__(data)
        self.window = window
        self.tolerance = tolerance
        self.name = "DoubleBottom"
        
    def _find_local_minima(self) -> List[int]:
        """
        국소 저점 찾기
        
        Returns:
            List[int]: 국소 저점의 인덱스 목록
        """
        # 국소 저점 찾기 (window/2 일 전후로 가장 낮은 가격)
        half_window = self.window // 2
        minima_indices = []
        
        # 데이터가 Series인지 확인하고 변환
        low_series = self.data['Low']
        if not isinstance(low_series, pd.Series):
            low_series = low_series.iloc[:, 0]
            
        for i in range(half_window, len(self.data) - half_window):
            # 윈도우 슬라이스 추출
            window_slice = low_series.iloc[i - half_window:i + half_window + 1]
            
            # 스칼라 값으로 변환
            window_min_value = float(window_slice.min())
            current_low_value = float(low_series.iloc[i])
            
            # 스칼라 값 비교 (정확한 부동 소수점 비교를 위해 근사값 비교 사용)
            if abs(current_low_value - window_min_value) < 1e-10:
                # 동일한 최소값이 여러 개 있는지 확인
                min_count = sum(abs(float(x) - window_min_value) < 1e-10 for x in window_slice)
                if min_count == 1:
                    minima_indices.append(i)
                    
        return minima_indices
    
    def detect(self) -> pd.DataFrame:
        """
        쌍바닥 패턴 감지
        
        Returns:
            pd.DataFrame: 패턴이 감지된 인덱스와 신호 강도가 포함된 데이터프레임
        """
        # 결과 데이터프레임 초기화
        result = pd.DataFrame(index=self.data.index)
        result['Pattern'] = None
        result['Strength'] = 0.0
        result['BottomIndex1'] = None
        result['BottomIndex2'] = None
        
        # 데이터 열 추출 및 Series 확인
        low_series = self.data['Low']
        high_series = self.data['High']
        close_series = self.data['Close']
        volume_series = self.data['Volume']
        
        if not isinstance(low_series, pd.Series):
            low_series = low_series.iloc[:, 0]
        if not isinstance(high_series, pd.Series):
            high_series = high_series.iloc[:, 0]
        if not isinstance(close_series, pd.Series):
            close_series = close_series.iloc[:, 0]
        if not isinstance(volume_series, pd.Series):
            volume_series = volume_series.iloc[:, 0]
        
        # 국소 저점 찾기
        minima_indices = self._find_local_minima()
        
        # 쌍바닥 패턴 감지
        for i in range(len(minima_indices) - 1):
            first_bottom_idx = minima_indices[i]
            
            # 두 번째 저점 찾기
            for j in range(i + 1, len(minima_indices)):
                second_bottom_idx = minima_indices[j]
                
                # 두 저점 간 간격 확인 (너무 가깝거나 너무 멀면 제외)
                time_diff = second_bottom_idx - first_bottom_idx
                if time_diff < self.window or time_diff > self.window * 3:
                    continue
                
                # 두 저점의 가격 차이 확인
                first_bottom_price = float(low_series.iloc[first_bottom_idx])
                second_bottom_price = float(low_series.iloc[second_bottom_idx])
                price_diff_pct = abs(second_bottom_price - first_bottom_price) / first_bottom_price
                
                # 두 저점의 가격이 유사한지 확인 (허용 오차 내)
                if price_diff_pct > self.tolerance:
                    continue
                
                # 두 저점 사이에 고점(반동) 확인
                mid_section = high_series.iloc[first_bottom_idx:second_bottom_idx]
                mid_high_price = float(mid_section.max())
                
                # 중간 고점이 저점보다 일정 비율 이상 높아야 함
                min_price = min(first_bottom_price, second_bottom_price)
                if (mid_high_price - min_price) / min_price < 0.05:
                    continue
                
                # 두 번째 저점 이후 상승 확인 (확인 거리)
                confirm_distance = min(10, len(self.data) - second_bottom_idx - 1)
                if confirm_distance <= 0:
                    continue
                
                confirm_price = float(close_series.iloc[second_bottom_idx + confirm_distance])
                second_bottom_to_confirm_pct = (confirm_price - second_bottom_price) / second_bottom_price
                
                # 상승 확인을 위한 최소 비율
                min_confirm_pct = 0.03
                
                # 두 번째 저점 이후 상승이 확인되면 패턴 감지
                if second_bottom_to_confirm_pct >= min_confirm_pct:
                    # 패턴 강도 계산 (여러 요소 고려)
                    # 1. 두 저점 가격의 유사성 (차이가 적을수록 강한 신호)
                    price_similarity = 1 - price_diff_pct / self.tolerance
                    
                    # 2. 두 번째 저점 이후 상승 강도
                    rise_strength = min(second_bottom_to_confirm_pct / 0.1, 1)
                    
                    # 3. 거래량 확인 (두 번째 저점에서 거래량 증가 시 더 강한 신호)
                    first_volume = float(volume_series.iloc[first_bottom_idx])
                    second_volume = float(volume_series.iloc[second_bottom_idx])
                    volume_increase = min(second_volume / first_volume if first_volume > 0 else 1, 2) / 2
                    
                    # 종합 강도 계산 (가중 평균)
                    strength = 0.4 * price_similarity + 0.4 * rise_strength + 0.2 * volume_increase
                    
                    # 결과 저장
                    for idx in range(second_bottom_idx, min(second_bottom_idx + confirm_distance + 1, len(self.data))):
                        result.loc[self.data.index[idx], 'Pattern'] = 'DoubleBottom'
                        result.loc[self.data.index[idx], 'Strength'] = strength
                        result.loc[self.data.index[idx], 'BottomIndex1'] = first_bottom_idx
                        result.loc[self.data.index[idx], 'BottomIndex2'] = second_bottom_idx
                        
        return result
    
    def get_signals(self) -> pd.DataFrame:
        """
        쌍바닥 패턴 기반 매매 신호 생성
        
        Returns:
            pd.DataFrame: 매매 신호가 포함된 데이터프레임
                - 1: 매수 신호
                - 0: 홀드 신호
                - -1: 매도 신호
        """
        try:
            # 패턴 감지
            patterns = self.detect()
            
            # 결과 데이터프레임 초기화
            signals = pd.DataFrame(index=self.data.index)
            signals['Signal'] = 0
            
            # 쌍바닥 패턴이 감지된 경우 매수 신호 생성
            pattern_mask = patterns['Pattern'] == 'DoubleBottom'
            
            # any() 메서드를 사용하여 마스크에 True가 있는지 확인
            if pattern_mask.any():
                pattern_indices = patterns[pattern_mask].index
                
                for idx in pattern_indices:
                    # 패턴 강도에 비례하는 매수 신호 생성
                    strength = float(patterns.loc[idx, 'Strength'])
                    signals.loc[idx, 'Signal'] = strength
                    
                    # 매수 신호 유지 (5일 동안)
                    # 날짜 인덱스인 경우와 정수 인덱스인 경우를 모두 처리
                    if isinstance(idx, pd.Timestamp):
                        for i in range(1, 6):
                            try:
                                next_idx = idx + pd.Timedelta(days=i)
                                if next_idx in signals.index:
                                    signals.loc[next_idx, 'Signal'] = max(0, strength - i * 0.1)
                            except Exception:
                                # 날짜 인덱스 처리 중 오류 발생 시 무시
                                pass
                    else:
                        # 정수 인덱스인 경우
                        for i in range(1, 6):
                            if idx + i < len(signals):
                                signals.iloc[idx + i, 0] = max(0, strength - i * 0.1)
            
            return signals
            
        except Exception as e:
            # 오류 발생 시 빈 신호 데이터프레임 반환
            print(f"쌍바닥 패턴 신호 생성 중 오류 발생: {e}")
            empty_signals = pd.DataFrame(index=self.data.index)
            empty_signals['Signal'] = 0
            return empty_signals 