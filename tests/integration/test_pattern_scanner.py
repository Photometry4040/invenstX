"""
패턴 스캐너 통합 테스트
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 필요한 함수 임포트
from main import scan_patterns, calculate_technical_indicators, plot_pattern_chart

class TestPatternScanner(unittest.TestCase):
    """패턴 스캐너 통합 테스트 클래스"""
    
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
        
        # 기술적 지표 계산
        self.test_data_with_indicators = calculate_technical_indicators(self.test_data)
    
    def test_scan_patterns_integration(self):
        """패턴 스캔 통합 테스트"""
        # 패턴 스캔 실행
        patterns_df = scan_patterns(self.test_data_with_indicators)
        
        # 결과가 None이 아닌지 확인
        self.assertIsNotNone(patterns_df)
        
        # 데이터프레임인지 확인
        self.assertIsInstance(patterns_df, pd.DataFrame)
        
        # 인덱스 길이가 원본 데이터와 동일한지 확인
        self.assertEqual(len(patterns_df.index), len(self.test_data.index))
        
        # 주요 컬럼이 포함되어 있는지 확인
        expected_columns = ['long_term', 'swing']
        for col in expected_columns:
            self.assertIn(col, patterns_df.columns)
    
    def test_plot_pattern_chart_integration(self):
        """패턴 차트 생성 통합 테스트"""
        # 패턴 스캔 실행
        patterns_df = scan_patterns(self.test_data_with_indicators)
        
        if patterns_df is not None:
            # 패턴 차트 생성
            chart = plot_pattern_chart(self.test_data, patterns_df, 'long_term')
            
            # 차트가 생성되었는지 확인
            self.assertIsNotNone(chart)
            
            # 적절한 객체 타입인지 확인
            import plotly.graph_objects as go
            self.assertIsInstance(chart, go.Figure)
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 빈 데이터프레임
        empty_df = pd.DataFrame()
        self.assertIsNone(scan_patterns(empty_df))
        
        # NaN 값이 있는 데이터
        data_with_nan = self.test_data.copy()
        data_with_nan.iloc[10:20, 0] = np.nan  # Open 데이터에 NaN 추가
        
        # NaN 값이 있는 데이터로 패턴 스캔
        patterns_with_nan = scan_patterns(data_with_nan)
        
        # NaN 값을 처리했는지 확인 (None이 아니거나 적절히 처리된 결과)
        if patterns_with_nan is not None:
            self.assertFalse(patterns_with_nan.isna().all().all())
        
        # 매우 작은 데이터셋
        small_df = self.test_data.iloc[:30]
        small_patterns = scan_patterns(small_df)
        self.assertIsNone(small_patterns)  # 50개 미만이면 None 반환
        
        # 필요한 열이 없는 데이터
        invalid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.assertIsNone(scan_patterns(invalid_df))

if __name__ == '__main__':
    unittest.main() 