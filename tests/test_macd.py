# tests/test_macd.py
"""
MACD 계산 기능 테스트
"""
import pandas as pd
from indicators.macd import calculate_macd

def test_calculate_macd():
    data = pd.DataFrame({"Close": [100, 102, 101, 103, 105]})
    result = calculate_macd(data)
    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert len(result["MACD"]) == 5