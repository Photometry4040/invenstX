#########################################
# 1) fetch_data.py (내장 예시)
#########################################
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, start_date, end_date):
    """
    yfinance를 사용하여 주식 데이터를 가져옴
    """
    try:
        # 티커 심볼 정리 (공백 및 특수문자 제거)
        ticker = ticker.strip().upper()
        
        logger.info(f"{ticker} 데이터를 {start_date}부터 {end_date}까지 가져오는 중")
        
        # yfinance Ticker 객체 생성
        stock = yf.Ticker(ticker)
        
        # 데이터 다운로드
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            logger.error("데이터가 비어 있습니다.")
            return None
            
        # 인덱스 타임존 제거
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        logger.info("데이터 가져오기 완료")
        return data
        
    except Exception as e:
        logger.error(f"데이터 가져오기 실패: {str(e)}")
        return None