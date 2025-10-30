"""
캔들스틱 패턴 감지 관련 기능을 제공하는 모듈
Doji, Hammer, Engulfing 등의 패턴을 감지
"""

import pandas as pd
import numpy as np
import logging
import traceback

def scan_patterns(data, min_points=5):
    """
    주식 데이터에서 캔들스틱 패턴을 감지합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        min_points (int): 패턴 감지에 필요한 최소 데이터 포인트
        
    Returns:
        pd.DataFrame: 패턴 감지 결과
    """
    # 데이터 유효성 검사
    if data is None or data.empty or len(data) < min_points:
        logging.error(f"패턴 감지 실패: 데이터가 부족합니다 ({len(data) if data is not None else 0}개 행, 최소 {min_points}개 필요)")
        # 빈 결과 반환
        pattern_df = pd.DataFrame(index=data.index if data is not None and not data.empty else [])
        for pattern in ['Doji', 'Hammer', 'Engulfing Bullish', 'Engulfing Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df
    
    # 필요한 컬럼 확인
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"패턴 감지 실패: 필요한 컬럼 누락 - {missing_columns}")
        # 빈 결과 반환
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing Bullish', 'Engulfing Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df
    
    try:
        # 결과를 저장할 데이터프레임 초기화 (모든 날짜에 대해 False로 설정)
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing Bullish', 'Engulfing Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        
        # 각 캔들에 대한 패턴 감지 계산
        for i in range(2, len(data)):
            # 현재, 이전 캔들의 데이터를 스칼라 값으로 추출
            open_curr = float(data['Open'].iloc[i])
            high_curr = float(data['High'].iloc[i])
            low_curr = float(data['Low'].iloc[i])
            close_curr = float(data['Close'].iloc[i])
            
            open_prev = float(data['Open'].iloc[i-1])
            high_prev = float(data['High'].iloc[i-1])
            low_prev = float(data['Low'].iloc[i-1])
            close_prev = float(data['Close'].iloc[i-1])
            
            curr_date = data.index[i]
            
            # 1. Doji 패턴 (시가와 종가가 거의 같음)
            body_size = abs(open_curr - close_curr)
            candle_range = high_curr - low_curr
            
            # 캔들 범위가 0보다 크고, 몸통 크기가 캔들 범위의 10% 미만인 경우 Doji로 판단
            if candle_range > 0 and body_size / candle_range < 0.1:
                pattern_df.at[data.index[i], 'Doji'] = True
                logging.info(f"Doji 패턴 감지: {curr_date.strftime('%Y-%m-%d')}")
            
            # 2. Hammer 패턴 (작은 몸통, 긴 아래 그림자)
            if close_curr > open_curr:  # 양봉
                body_size = close_curr - open_curr
                upper_shadow = high_curr - close_curr
                lower_shadow = open_curr - low_curr
            else:  # 음봉
                body_size = open_curr - close_curr
                upper_shadow = high_curr - open_curr
                lower_shadow = close_curr - low_curr
            
            # 캔들 범위가 0보다 크고, 몸통이 작으며, 아래 그림자가 길고, 위 그림자가 짧은 경우 Hammer로 판단
            if (candle_range > 0 and 
                body_size / candle_range < 0.3 and 
                lower_shadow > 2 * body_size and 
                upper_shadow < 0.3 * lower_shadow):
                pattern_df.at[data.index[i], 'Hammer'] = True
                logging.info(f"Hammer 패턴 감지: {curr_date.strftime('%Y-%m-%d')}")
            
            # 3. Engulfing 패턴 (이전 캔들을 완전히 감싸는 패턴)
            # 3.1 Bullish Engulfing (이전 캔들이 음봉, 현재 캔들이 양봉이며 이전 캔들을 완전히 감싸는 경우)
            if (close_prev < open_prev and  # 이전 캔들이 음봉
                close_curr > open_curr and  # 현재 캔들이 양봉
                open_curr <= close_prev and  # 현재 시가가 이전 종가보다 낮거나 같음
                close_curr >= open_prev):  # 현재 종가가 이전 시가보다 높거나 같음
                pattern_df.at[data.index[i], 'Engulfing Bullish'] = True
                logging.info(f"Bullish Engulfing 패턴 감지: {curr_date.strftime('%Y-%m-%d')}")
            
            # 3.2 Bearish Engulfing (이전 캔들이 양봉, 현재 캔들이 음봉이며 이전 캔들을 완전히 감싸는 경우)
            elif (close_prev > open_prev and  # 이전 캔들이 양봉
                  close_curr < open_curr and  # 현재 캔들이 음봉
                  open_curr >= close_prev and  # 현재 시가가 이전 종가보다 높거나 같음
                  close_curr <= open_prev):  # 현재 종가가 이전 시가보다 낮거나 같음
                pattern_df.at[data.index[i], 'Engulfing Bearish'] = True
                logging.info(f"Bearish Engulfing 패턴 감지: {curr_date.strftime('%Y-%m-%d')}")
        
        # 상승/하락 추세 감지 - 간단한 방식으로 변경
        # 최소 10일 이상의 데이터가 있을 때만 추세 계산
        if len(data) >= 10:
            for i in range(10, len(data)):
                curr_date = data.index[i]
                # 10일 전보다 종가가 높고, 10일 평균보다 높으면 상승 추세
                curr_close = float(data['Close'].iloc[i])
                prev_close = float(data['Close'].iloc[i-10])
                avg_close = float(data['Close'].iloc[i-10:i].mean())
                
                if curr_close > prev_close and curr_close > avg_close:
                    pattern_df.at[data.index[i], 'Uptrend'] = True
                
                # 10일 전보다 종가가 낮고, 10일 평균보다 낮으면 하락 추세
                elif curr_close < prev_close and curr_close < avg_close:
                    pattern_df.at[data.index[i], 'Downtrend'] = True
        
        # 패턴 감지 결과 요약 로그
        pattern_counts = {col: pattern_df[col].sum() for col in pattern_df.columns}
        logging.info(f"패턴 감지 완료: {pattern_counts}")
        return pattern_df
    
    except Exception as e:
        logging.error(f"패턴 감지 중 오류: {str(e)}")
        logging.debug(traceback.format_exc())
        
        # 오류 발생 시 기본 결과 반환
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing Bullish', 'Engulfing Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df 