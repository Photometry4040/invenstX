"""
주식 차트 패턴 분석기

다양한 차트 패턴을 인식하고 분석하는 기본 클래스입니다.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

class Pattern(ABC):
    """
    차트 패턴의 추상 기본 클래스
    
    모든 패턴 인식 클래스는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        패턴 클래스 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("데이터프레임에는 'Open', 'High', 'Low', 'Close', 'Volume' 열이 포함되어야 합니다.")
        
        self.data = data.copy()
        self.name = self.__class__.__name__
        
    @abstractmethod
    def detect(self) -> pd.DataFrame:
        """
        데이터에서 패턴을 감지합니다.
        
        Returns:
            pd.DataFrame: 패턴이 감지된 인덱스와 신호 강도가 포함된 데이터프레임
        """
        pass
    
    @abstractmethod
    def get_signals(self) -> pd.DataFrame:
        """
        패턴 기반 매매 신호를 생성합니다.
        
        Returns:
            pd.DataFrame: 매매 신호가 포함된 데이터프레임
                - 1: 매수 신호
                - 0: 홀드 신호
                - -1: 매도 신호
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} Pattern Analyzer"


class ChartPattern(Pattern):
    """
    차트 패턴 기본 클래스
    
    Pattern 클래스를 상속받아 구현한 기본 차트 패턴 클래스입니다.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        차트 패턴 클래스 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
        """
        super().__init__(data)
    
    def detect(self) -> pd.Series:
        """
        데이터에서 패턴을 감지합니다.
        
        Returns:
            pd.Series: 패턴이 감지된 인덱스와 신호 값이 포함된 시리즈
        """
        # 기본 구현은 빈 시리즈 반환
        return pd.Series(0, index=self.data.index)
    
    def get_signals(self, style='default') -> pd.DataFrame:
        """
        패턴 기반 매매 신호를 생성합니다.
        
        Args:
            style (str): 투자 스타일 ('default', 'long_term', 'swing')
            
        Returns:
            pd.DataFrame: 매매 신호가 포함된 데이터프레임
                - Signal: 매매 신호 (-1, 0, 1)
                - Strength: 신호 강도 (0.0 ~ 1.0)
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
                    
                    # 기본 강도 설정
                    strength_value = abs(float(pattern_result.iloc[i]))
                    signals.iloc[i, signals.columns.get_loc('Strength')] = strength_value
            
            return signals
            
        except Exception as e:
            # 오류 발생 시 빈 신호 데이터프레임 반환
            print(f"차트 패턴 신호 생성 중 오류 발생: {e}")
            empty_signals = pd.DataFrame(index=self.data.index)
            empty_signals['Signal'] = 0
            empty_signals['Strength'] = 0.0
            return empty_signals


class PatternAnalyzer:
    """
    여러 패턴을 결합하여 종합적인 분석을 제공하는 클래스
    """
    
    def __init__(self, data: pd.DataFrame, patterns: Optional[List[Pattern]] = None):
        """
        패턴 분석기 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            patterns (List[Pattern], optional): 분석에 사용할 패턴 클래스 목록
        """
        self.data = data.copy()
        self.patterns = patterns or []
        self.signals = pd.DataFrame(index=data.index)
        self.signals['Combined'] = 0
        
    def add_pattern(self, pattern: Pattern) -> None:
        """
        분석에 패턴을 추가합니다.
        
        Args:
            pattern (Pattern): 추가할 패턴 클래스 인스턴스
        """
        self.patterns.append(pattern)
        
    def analyze(self, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        모든 패턴을 분석하고 결합된 신호를 생성합니다.
        
        Args:
            weights (Dict[str, float], optional): 각 패턴에 적용할 가중치
                예: {'BollingerBands': 1.0, 'DoubleBottom': 0.8}
                
        Returns:
            pd.DataFrame: 결합된 신호가 포함된 데이터프레임
        """
        if not self.patterns:
            logging.warning("패턴이 지정되지 않았습니다. 먼저 add_pattern() 메서드를 사용하여 패턴을 추가하세요.")
            return self.signals
        
        # 기본 가중치 설정
        if weights is None:
            weights = {pattern.name: 1.0 for pattern in self.patterns}
        
        # 각 패턴에서 신호 가져오기
        for pattern in self.patterns:
            try:
                pattern_signals = pattern.get_signals()
                
                # 신호 열이 있는지 확인
                if 'Signal' in pattern_signals.columns:
                    self.signals[pattern.name] = pattern_signals['Signal']
                    
                    # 가중치 적용
                    weight = weights.get(pattern.name, 1.0)
                    self.signals['Combined'] += pattern_signals['Signal'] * weight
            except Exception as e:
                logging.error(f"{pattern.name} 패턴 신호 처리 중 오류 발생: {e}")
        
        # Combined 신호 정규화 (-1 ~ 1 사이)
        max_abs = max(abs(self.signals['Combined'].max()), abs(self.signals['Combined'].min()))
        if max_abs > 0:
            self.signals['Combined'] = self.signals['Combined'] / max_abs
            
        # 최종 매매 신호 생성 (임계값 적용)
        self.signals['Signal'] = 0
        self.signals.loc[self.signals['Combined'] >= 0.5, 'Signal'] = 1  # 매수 신호
        self.signals.loc[self.signals['Combined'] <= -0.5, 'Signal'] = -1  # 매도 신호
        
        # Strength 열 추가
        self.signals['Strength'] = self.signals['Combined'].abs()
        
        return self.signals
    
    def get_investment_style_signals(self, style: str = 'long_term') -> pd.DataFrame:
        """
        투자 스타일에 맞는 신호를 생성합니다.
        
        Args:
            style (str): 투자 스타일 ('long_term', 'swing_trade', 'default')
            
        Returns:
            pd.DataFrame: 투자 스타일에 맞는 신호가 포함된 데이터프레임
        """
        # 투자 스타일별 가중치 설정
        style_weights = {
            'long_term': {
                'BollingerBands': 0.7,
                'DoubleBottom': 0.9,
                'DoubleTop': 0.9,
                'HeadAndShoulders': 0.8,
            },
            'swing_trade': {
                'BollingerBands': 1.0,
                'DoubleBottom': 0.8,
                'DoubleTop': 0.8,
                'HeadAndShoulders': 0.7,
            },
            'default': {
                'BollingerBands': 1.0,
                'DoubleBottom': 1.0,
                'DoubleTop': 1.0,
                'HeadAndShoulders': 1.0,
            }
        }
        
        # 투자 스타일에 맞는 가중치 선택
        if style not in style_weights:
            logging.warning(f"지정한 투자 스타일 '{style}'이 유효하지 않습니다. 기본 스타일을 사용합니다.")
            style = 'default'
            
        weights = style_weights[style]
        
        # 신호 분석
        signals = self.analyze(weights=weights)
        
        # 투자 스타일에 맞게 신호 필터링
        filtered_signals = signals.copy()
        
        if style == 'long_term':
            # 장기 투자: 매수 신호는 유지하되, 작은 매도 신호는 무시
            # (임계값 높임)
            filtered_signals['Signal'] = 0
            filtered_signals.loc[signals['Combined'] >= 0.6, 'Signal'] = 1
            filtered_signals.loc[signals['Combined'] <= -0.7, 'Signal'] = -1
            
            # 장기 투자에서는 매수 신호가 더 오래 유지되도록 함
            # (매수 신호 후 5일간 홀드 신호가 나오더라도 매수 상태 유지)
            buy_signals = filtered_signals['Signal'] == 1
            for i in range(1, 6):
                if i < len(filtered_signals):
                    mask = buy_signals.shift(i, fill_value=False)
                    filtered_signals.loc[mask & (filtered_signals['Signal'] == 0), 'Signal'] = 1
                    
        elif style == 'swing_trade':
            # 스윙 트레이드: 중간 정도의 임계값 적용
            filtered_signals['Signal'] = 0
            filtered_signals.loc[signals['Combined'] >= 0.5, 'Signal'] = 1
            filtered_signals.loc[signals['Combined'] <= -0.5, 'Signal'] = -1
            
            # 불필요한 빈번한 매매 신호 제거 (3일 이내 동일 신호 반복 방지)
            for i in range(1, 3):
                if i < len(filtered_signals):
                    # 이전 신호와 현재 신호가 같으면 현재 신호를 0으로 설정
                    same_signal = filtered_signals['Signal'].shift(i) == filtered_signals['Signal']
                    filtered_signals.loc[same_signal, 'Signal'] = 0
        
        return filtered_signals
            
    def __str__(self) -> str:
        pattern_names = [pattern.name for pattern in self.patterns]
        return f"Pattern Analyzer with {len(pattern_names)} patterns: {', '.join(pattern_names)}" 