"""
세션 상태를 활용한 데이터 캐싱 관리 유틸리티
"""

import streamlit as st
import pandas as pd
from datetime import datetime

def get_cache_key(ticker, start_date, end_date):
    """
    캐시 키를 생성합니다.
    
    Args:
        ticker (str): 주식 티커
        start_date: 시작 날짜
        end_date: 종료 날짜
        
    Returns:
        str: 캐시 키
    """
    start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)
    return f"{ticker}_{start_str}_{end_str}"

def get_cached_data(ticker, start_date, end_date, load_func):
    """
    캐시된 데이터를 가져오거나, 없으면 로드하여 캐시에 저장합니다.
    
    Args:
        ticker (str): 주식 티커
        start_date: 시작 날짜
        end_date: 종료 날짜
        load_func (callable): 데이터 로드 함수
        
    Returns:
        pandas.DataFrame: 주식 데이터
    """
    # 세션 스테이트 초기화
    if 'cache_keys' not in st.session_state:
        st.session_state.cache_keys = {}
        
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    
    cache_key = get_cache_key(ticker, start_date, end_date)
    
    # 캐시에 있으면 반환
    if cache_key in st.session_state.data_cache:
        st.info(f"캐시된 {ticker} 데이터를 사용합니다.")
        return st.session_state.data_cache[cache_key]
    
    # 없으면 로드하여 캐시에 저장
    data = load_func(ticker, start_date, end_date)
    if data is not None:
        st.session_state.data_cache[cache_key] = data
        st.session_state.cache_keys[ticker] = cache_key
        st.success(f"{ticker} 데이터를 성공적으로 불러왔습니다.")
    else:
        st.error(f"{ticker} 데이터를 불러오지 못했습니다.")
    
    return data

def clear_cache(ticker=None):
    """
    특정 티커 또는 모든 캐시를 제거합니다.
    
    Args:
        ticker (str, optional): 제거할 티커. None이면 모든 캐시 제거.
    """
    if 'data_cache' not in st.session_state:
        return
        
    if ticker is None:
        # 모든 캐시 제거
        st.session_state.data_cache = {}
        st.session_state.cache_keys = {}
        st.info("모든 캐시가 제거되었습니다.")
    elif 'cache_keys' in st.session_state and ticker in st.session_state.cache_keys:
        # 특정 티커 캐시 제거
        cache_key = st.session_state.cache_keys[ticker]
        if cache_key in st.session_state.data_cache:
            del st.session_state.data_cache[cache_key]
        del st.session_state.cache_keys[ticker]
        st.info(f"{ticker} 캐시가 제거되었습니다.") 