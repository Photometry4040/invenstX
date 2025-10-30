# indicators/macd.py
"""
MACD(이동평균 수렴 확산 지표)를 계산하는 모듈
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """
    주식 데이터에 MACD 계산 및 추가
    
    :param data: 주식 데이터프레임 (Close 열 포함)
    :param short_period: 단기 이동평균 기간
    :param long_period: 장기 이동평균 기간
    :param signal_period: 시그널 라인 기간
    :return: MACD가 추가된 데이터프레임
    """
    try:
        short_ema = data["Close"].ewm(span=short_period, adjust=False).mean()
        long_ema = data["Close"].ewm(span=long_period, adjust=False).mean()
        data["MACD"] = short_ema - long_ema
        data["MACD_signal"] = data["MACD"].ewm(span=signal_period, adjust=False).mean()
        logging.info("MACD calculated")
        return data
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        return data