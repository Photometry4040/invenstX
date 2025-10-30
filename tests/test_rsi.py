# tests/test_rsi.py
"""
RSI 계산 기능 테스트
"""
import pandas as pd
from indicators.rsi import calculate_rsi

def test_calculate_rsi():
    data = pd.DataFrame({"Close": [100, 102, 101, 103, 105]})
    result = calculate_rsi(data, window=3)
    assert "RSI" in result.columns
    assert len(result["RSI"]) == 5