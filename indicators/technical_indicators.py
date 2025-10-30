"""
주식 데이터에 대한 기술적 지표를 계산하는 함수 모음
MACD, RSI, 볼린저 밴드, ATR 등 다양한 기술적 지표 계산
"""

import pandas as pd
import numpy as np
import logging
import traceback

def calculate_technical_indicators(data):
    """
    기본적인 기술적 지표 계산 (MA5, MA20, MA60, RSI, MACD)
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        
    Returns:
        pd.DataFrame: 기술적 지표가 추가된 데이터 또는 None (오류 발생 시)
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("기술적 지표 계산 실패: 빈 데이터")
            return None

        # 필요한 컬럼 확인
        required_columns = ['Close']
        if not all(col in data.columns for col in required_columns):
            logging.error(f"기술적 지표 계산 실패: 필요한 컬럼 누락 - {required_columns}")
            return None
            
        # 데이터 복사본 생성
        data_copy = data.copy()
        
        # 이동평균선
        data_copy['MA5'] = data_copy['Close'].rolling(window=5).mean()
        data_copy['MA20'] = data_copy['Close'].rolling(window=20).mean()
        data_copy['MA60'] = data_copy['Close'].rolling(window=60).mean()
        
        # RSI
        delta = data_copy['Close'].diff()
        delta = delta.fillna(0)  # 첫 번째 NaN 값 처리
        
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # 0으로 나누는 것 방지
        loss_eps = loss.replace(0, np.finfo(float).eps)
        rs = gain / loss_eps
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data_copy['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data_copy['Close'].ewm(span=26, adjust=False).mean()
        data_copy['MACD'] = exp1 - exp2
        data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
        data_copy['MACD_Hist'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
        logging.info("기본 기술적 지표 계산 완료")
        return data_copy
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        logging.debug(traceback.format_exc())
        return None

def add_technical_indicators(data):
    """
    주식 데이터에 다양한 기술적 지표를 추가하는 함수
    (MA, RSI, MACD, 볼린저 밴드, ATR, OBV, 스토캐스틱 등)
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        
    Returns:
        pd.DataFrame: 기술적 지표가 추가된 데이터
    """
    # 입력 데이터 유효성 검사
    if data is None or data.empty:
        logging.error("기술적 지표 추가 실패: 빈 데이터")
        return data

    # 필요한 컬럼 확인
    required_columns = ['Close', 'High', 'Low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    # 'Volume'은 있으면 좋지만 없으면 더미 데이터로 대체
    if 'Volume' not in data.columns and len(missing_columns) == 0:
        logging.warning("Volume 데이터가 없습니다. 더미 데이터로 대체합니다.")
        data['Volume'] = 0  # 더미 데이터

    # 필수 컬럼이 없으면 오류 로그 남기고 원본 반환
    if missing_columns:
        logging.error(f"기술적 지표 추가 실패: 필요한 컬럼 누락 - {missing_columns}")
        return data
    
    try:
        # 데이터 복사본 생성
        data_copy = data.copy()
        
        # 데이터 타입 검사 및 변환
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in data_copy.columns:
                try:
                    # 숫자형이 아닌 경우에만 변환 시도
                    if not pd.api.types.is_numeric_dtype(data_copy[col]):
                        logging.warning(f"{col} 열이 숫자형이 아닙니다. 현재 타입: {data_copy[col].dtype}")
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                        logging.info(f"{col} 열을 숫자형으로 변환했습니다.")
                except Exception as e:
                    logging.error(f"{col} 열을 숫자형으로 변환하는 중 오류 발생: {str(e)}")
                    logging.debug(f"{col} 열의 처음 5개 값: {data_copy[col].head().tolist() if col in data_copy.columns else 'column not found'}")
        
        # NaN 값 확인
        for col in [c for c in numeric_columns if c in data_copy.columns]:
            nan_count = data_copy[col].isna().sum()
            if nan_count > 0:
                logging.warning(f"{col} 열에 {nan_count}개의 NaN 값이 있습니다.")
                
                # 종가, 고가, 저가의 NaN 값이 너무 많으면 처리 중단
                if col in ['Close', 'High', 'Low'] and nan_count > len(data_copy) * 0.1:  # 10% 이상이 NaN
                    logging.error(f"{col} 열의 NaN 값이 너무 많습니다 ({nan_count}/{len(data_copy)}). 기술적 지표 계산을 중단합니다.")
                    return data  # 원본 데이터 반환
        
        # MA 계산 전 NaN 값 제거 (종가에 NaN이 있는 경우 제거)
        if data_copy['Close'].isna().any():
            logging.warning(f"종가에 NaN 값이 있습니다. 보간을 시도합니다.")
            # 앞뒤 값으로 보간 시도
            data_copy['Close'] = data_copy['Close'].interpolate(method='linear', limit_direction='both')
            
            # 보간 후에도 NaN이 남아있는지 확인
            if data_copy['Close'].isna().any():
                logging.error("보간 후에도 종가에 NaN 값이 남아있습니다. 계산을 중단합니다.")
                return data  # 원본 데이터 반환
        
        # 이동평균 계산
        data_copy['MA5'] = data_copy['Close'].rolling(window=5).mean()
        data_copy['MA10'] = data_copy['Close'].rolling(window=10).mean()
        data_copy['MA20'] = data_copy['Close'].rolling(window=20).mean()
        data_copy['MA50'] = data_copy['Close'].rolling(window=50).mean()
        data_copy['MA200'] = data_copy['Close'].rolling(window=200).mean()
        
        # RSI 계산 (14일)
        delta = data_copy['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # 0으로 나누기 방지
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD 계산
        exp1 = data_copy['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data_copy['Close'].ewm(span=26, adjust=False).mean()
        data_copy['MACD'] = exp1 - exp2
        data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
        data_copy['MACD_Hist'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
        # 볼린저 밴드 (20일)
        data_copy['BB_Middle'] = data_copy['Close'].rolling(window=20).mean()
        data_copy['BB_STD'] = data_copy['Close'].rolling(window=20).std()
        data_copy['BB_Upper'] = data_copy['BB_Middle'] + (data_copy['BB_STD'] * 2)
        data_copy['BB_Lower'] = data_copy['BB_Middle'] - (data_copy['BB_STD'] * 2)
        
        # ATR 계산 (14일)
        high_low = data_copy['High'] - data_copy['Low']
        high_close = abs(data_copy['High'] - data_copy['Close'].shift())
        low_close = abs(data_copy['Low'] - data_copy['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data_copy['ATR'] = tr.rolling(window=14).mean()
        
        # OBV (On-Balance Volume)
        if 'Volume' in data_copy.columns:
            obv = [0]
            for i in range(1, len(data_copy)):
                if data_copy['Close'].iloc[i] > data_copy['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data_copy['Volume'].iloc[i])
                elif data_copy['Close'].iloc[i] < data_copy['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data_copy['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            data_copy['OBV'] = obv
        
        # 스토캐스틱 오실레이터 (14일)
        low_14 = data_copy['Low'].rolling(window=14).min()
        high_14 = data_copy['High'].rolling(window=14).max()
        data_copy['Stoch_K'] = 100 * ((data_copy['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        data_copy['Stoch_D'] = data_copy['Stoch_K'].rolling(window=3).mean()
        
        logging.info("다양한 기술적 지표가 성공적으로 추가되었습니다.")
        return data_copy
    
    except Exception as e:
        logging.error(f"기술적 지표 추가 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        # 기존 데이터 반환
        return data


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    볼린저 밴드 계산
    
    Args:
        data (pd.DataFrame): 주식 데이터
        window (int): 이동평균 기간
        num_std (int): 표준편차 승수
        
    Returns:
        pd.DataFrame: 볼린저 밴드가 추가된 데이터
    """
    try:
        if 'Close' not in data.columns:
            logging.error("볼린저 밴드 계산 실패: Close 컬럼 없음")
            return data
            
        # 데이터 복사본 생성
        data_copy = data.copy()
        
        # 중간 밴드 (단순 이동평균)
        data_copy['BB_Middle'] = data_copy['Close'].rolling(window=window).mean()
        
        # 표준편차 계산
        data_copy['BB_STD'] = data_copy['Close'].rolling(window=window).std()
        
        # 상단 및 하단 밴드
        data_copy['BB_Upper'] = data_copy['BB_Middle'] + (data_copy['BB_STD'] * num_std)
        data_copy['BB_Lower'] = data_copy['BB_Middle'] - (data_copy['BB_STD'] * num_std)
        
        # 밴드 폭
        data_copy['BB_Width'] = (data_copy['BB_Upper'] - data_copy['BB_Lower']) / data_copy['BB_Middle']
        
        # %B 지표 (현재 가격이 밴드 내에서 어느 위치에 있는지)
        data_copy['BB_PercentB'] = (data_copy['Close'] - data_copy['BB_Lower']) / (data_copy['BB_Upper'] - data_copy['BB_Lower'])
        
        return data_copy
    except Exception as e:
        logging.error(f"볼린저 밴드 계산 중 오류: {str(e)}")
        return data


def calculate_rsi(data, window=14):
    """
    RSI(Relative Strength Index) 계산
    
    Args:
        data (pd.DataFrame): 주식 데이터
        window (int): RSI 계산 기간
        
    Returns:
        pd.DataFrame: RSI가 추가된 데이터
    """
    try:
        if 'Close' not in data.columns:
            logging.error("RSI 계산 실패: Close 컬럼 없음")
            return data
            
        # 데이터 복사본 생성
        data_copy = data.copy()
        
        # 가격 변화량
        delta = data_copy['Close'].diff()
        
        # 상승/하락 구분
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 평균 상승/하락 계산
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # RS 계산 (0으로 나누기 방지)
        rs = avg_gain / (avg_loss + 1e-10)
        
        # RSI 계산
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        return data_copy
    except Exception as e:
        logging.error(f"RSI 계산 중 오류: {str(e)}")
        return data


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    MACD(Moving Average Convergence Divergence) 계산
    
    Args:
        data (pd.DataFrame): 주식 데이터
        fast (int): 빠른 EMA 기간
        slow (int): 느린 EMA 기간
        signal (int): 시그널 EMA 기간
        
    Returns:
        pd.DataFrame: MACD가 추가된 데이터
    """
    try:
        if 'Close' not in data.columns:
            logging.error("MACD 계산 실패: Close 컬럼 없음")
            return data
            
        # 데이터 복사본 생성
        data_copy = data.copy()
        
        # 빠른 EMA와 느린 EMA 계산
        fast_ema = data_copy['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = data_copy['Close'].ewm(span=slow, adjust=False).mean()
        
        # MACD 라인
        data_copy['MACD'] = fast_ema - slow_ema
        
        # 시그널 라인
        data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=signal, adjust=False).mean()
        
        # MACD 히스토그램
        data_copy['MACD_Hist'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
        return data_copy
    except Exception as e:
        logging.error(f"MACD 계산 중 오류: {str(e)}")
        return data 