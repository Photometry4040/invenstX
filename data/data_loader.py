"""
주식 데이터 로드 관련 함수
다양한 소스에서 주식 데이터를 가져오고 전처리하는 함수 제공
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import traceback
from datetime import datetime, timedelta
import pytz

def load_stock_data(ticker, start_date=None, end_date=None, interval='1d'):
    """
    Yahoo Finance API를 통해 주식 데이터 로드
    
    Args:
        ticker (str): 종목 코드 (예: 'AAPL', '005930.KS')
        start_date (str, optional): 시작 날짜 ('YYYY-MM-DD' 형식)
        end_date (str, optional): 종료 날짜 ('YYYY-MM-DD' 형식)
        interval (str, optional): 데이터 간격 ('1d', '1wk', '1mo' 등)
        
    Returns:
        pd.DataFrame: OHLCV 형식의 주식 데이터 또는 None (오류 발생 시)
    """
    try:
        # 날짜 형식 검증
        if start_date and end_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                logging.error("날짜 형식이 올바르지 않습니다. 'YYYY-MM-DD' 형식이어야 합니다.")
                return None
        
        # 날짜 설정
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 티커 형식 검증
        ticker = ticker.upper().strip()
        
        # 한국 주식은 .KS 또는 .KQ 접미사 추가
        if ticker.isdigit() and len(ticker) == 6:
            ticker = f"{ticker}.KS"  # 기본적으로 KOSPI로 가정
        
        logging.info(f"{ticker} 데이터 로드 중... (기간: {start_date} ~ {end_date}, 간격: {interval})")
        
        # Yahoo Finance API를 통해 데이터 로드
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        # 데이터 검증
        if stock_data.empty:
            logging.warning(f"{ticker} 데이터를 찾을 수 없습니다.")
            return None
        
        # 컬럼명 표준화
        stock_data = stock_data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        })
        
        # 인덱스가 datetime 형식인지 확인
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            logging.warning("인덱스가 datetime 형식이 아닙니다. datetime 형식으로 변환합니다.")
            stock_data.index = pd.to_datetime(stock_data.index)
        
        # 결측치 처리
        na_count = stock_data.isna().sum().sum()
        if na_count > 0:
            logging.warning(f"데이터에 {na_count}개의 결측치가 있습니다.")
            
            # Volume의 NaN 값은 0으로 대체
            if 'Volume' in stock_data.columns and stock_data['Volume'].isna().any():
                stock_data['Volume'] = stock_data['Volume'].fillna(0)
            
            # 가격 데이터의 결측치는 전방향 채우기
            price_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
            for col in price_cols:
                if col in stock_data.columns and stock_data[col].isna().any():
                    stock_data[col] = stock_data[col].fillna(method='ffill')
        
        # 데이터 날짜 역순 정렬 (최신 데이터가 마지막에 오도록)
        stock_data = stock_data.sort_index()
        
        logging.info(f"{ticker} 데이터 로드 성공: {len(stock_data)}개 행")
        return stock_data
    
    except Exception as e:
        logging.error(f"주식 데이터 로드 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        return None


def load_multiple_stocks(tickers, start_date=None, end_date=None, interval='1d'):
    """
    여러 종목의 주식 데이터를 동시에 로드
    
    Args:
        tickers (list): 종목 코드 리스트
        start_date (str, optional): 시작 날짜 ('YYYY-MM-DD' 형식)
        end_date (str, optional): 종료 날짜 ('YYYY-MM-DD' 형식)
        interval (str, optional): 데이터 간격 ('1d', '1wk', '1mo' 등)
        
    Returns:
        dict: {ticker: pd.DataFrame} 형식의 주식 데이터 딕셔너리
    """
    result = {}
    for ticker in tickers:
        data = load_stock_data(ticker, start_date, end_date, interval)
        if data is not None:
            result[ticker] = data
    
    return result


def get_latest_market_data(market='KS11', days=1):
    """
    최신 시장 데이터 로드 (KOSPI 또는 KOSDAQ 지수)
    
    Args:
        market (str): 시장 코드 ('KS11': KOSPI, 'KQ11': KOSDAQ)
        days (int): 가져올 데이터 일수
        
    Returns:
        pd.DataFrame: 시장 데이터
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        market_symbol = '^KS11' if market == 'KS11' else '^KQ11'  # KOSPI 또는 KOSDAQ
        
        market_data = yf.download(market_symbol, 
                                  start=start_date.strftime('%Y-%m-%d'),
                                  end=end_date.strftime('%Y-%m-%d'))
        
        if market_data.empty:
            logging.warning(f"{market} 시장 데이터를 찾을 수 없습니다.")
            return None
            
        return market_data
    
    except Exception as e:
        logging.error(f"시장 데이터 로드 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        return None 