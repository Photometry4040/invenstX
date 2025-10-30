import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rsi(data, window):
    logging.info(f"RSI 계산 중 (window: {window})")
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    logging.info("RSI 계산 완료")
    return data