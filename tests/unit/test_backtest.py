"""
백테스팅 기능 단위 테스트
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 백테스팅 함수 임포트
from main import backtest_ma_crossover, backtest_rsi_strategy, backtest_bollinger_bands

class TestBacktestFunctions(unittest.TestCase):
    """백테스팅 함수 테스트 클래스"""
    
    def setUp(self):
        """테스트 데이터 설정"""
        # 테스트용 데이터 생성
        dates = pd.date_range(start='2020-01-01', end='2020-12-31')
        n = len(dates)
        
        # 간단한 시뮬레이션 데이터 생성
        np.random.seed(42)  # 재현성을 위한 시드 설정
        
        # 랜덤한 트렌드 생성
        trend = np.cumsum(np.random.normal(0.001, 0.01, n))
        
        # 가격 데이터 생성
        close_price = 100 * (1 + trend)
        high_price = close_price * (1 + np.random.uniform(0, 0.03, n))
        low_price = close_price * (1 - np.random.uniform(0, 0.03, n))
        open_price = low_price + np.random.uniform(0, 1, n) * (high_price - low_price)
        
        # 거래량 데이터
        volume = np.random.uniform(1000, 5000, n)
        
        # DataFrame 생성
        self.test_data = pd.DataFrame({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }, index=dates)
    
    def test_ma_crossover_strategy(self):
        """이동평균 교차 전략 테스트"""
        result = backtest_ma_crossover(self.test_data, 10, 30, 10000)
        
        # 결과가 None이 아닌지 확인
        self.assertIsNotNone(result)
        
        # 필요한 키가 결과에 포함되어 있는지 확인
        self.assertIn('final_value', result)
        self.assertIn('returns', result)
        self.assertIn('portfolio_values', result)
        self.assertIn('trades', result)
        
        # 포트폴리오 가치가 양수인지 확인
        self.assertGreater(result['final_value'], 0)
    
    def test_rsi_strategy(self):
        """RSI 전략 테스트"""
        result = backtest_rsi_strategy(self.test_data, 14, 30, 70, 10000)
        
        # 결과가 None이 아닌지 확인
        self.assertIsNotNone(result)
        
        # 필요한 키가 결과에 포함되어 있는지 확인
        self.assertIn('final_value', result)
        self.assertIn('returns', result)
        self.assertIn('portfolio_values', result)
        self.assertIn('trades', result)
        
        # 포트폴리오 가치가 양수인지 확인
        self.assertGreater(result['final_value'], 0)
    
    def test_bollinger_bands_strategy(self):
        """볼린저 밴드 전략 테스트"""
        result = backtest_bollinger_bands(self.test_data, 20, 2.0, 10000)
        
        # 결과가 None이 아닌지 확인
        self.assertIsNotNone(result)
        
        # 필요한 키가 결과에 포함되어 있는지 확인
        self.assertIn('final_value', result)
        self.assertIn('returns', result)
        self.assertIn('portfolio_values', result)
        self.assertIn('trades', result)
        
        # 포트폴리오 가치가 양수인지 확인
        self.assertGreater(result['final_value'], 0)
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 빈 데이터프레임
        empty_df = pd.DataFrame()
        self.assertIsNone(backtest_ma_crossover(empty_df, 10, 30, 10000))
        self.assertIsNone(backtest_rsi_strategy(empty_df, 14, 30, 70, 10000))
        self.assertIsNone(backtest_bollinger_bands(empty_df, 20, 2.0, 10000))
        
        # 데이터가 적은 경우
        small_df = self.test_data.iloc[:5]
        # 결과가 None이거나 처리될 수 있어야 함
        ma_result = backtest_ma_crossover(small_df, 10, 30, 10000)
        rsi_result = backtest_rsi_strategy(small_df, 14, 30, 70, 10000)
        bb_result = backtest_bollinger_bands(small_df, 20, 2.0, 10000)
        
        # 초기 자본이 0이거나 음수인 경우
        self.assertIsNotNone(backtest_ma_crossover(self.test_data, 10, 30, 0))
        self.assertIsNotNone(backtest_rsi_strategy(self.test_data, 14, 30, 70, 0))
        self.assertIsNotNone(backtest_bollinger_bands(self.test_data, 20, 2.0, 0))

if __name__ == '__main__':
    unittest.main() 