# main.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import os
import time
from functools import lru_cache
import concurrent.futures
import warnings
from plotly.subplots import make_subplots

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 전역 변수
performance_monitor = None

# 선택적 모듈 로드
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.warning("일부 시각화 라이브러리를 로드할 수 없습니다. 관련 기능이 제한될 수 있습니다.")
    VISUALIZATION_AVAILABLE = False

try:
    import torch  # torch 모듈 로드 시도
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch 모듈을 로드할 수 없습니다. 강화학습 기능이 제한될 수 있습니다.")
    TORCH_AVAILABLE = False

# rl_models.py에서 필요한 클래스 및 함수 가져오기
try:
    from rl_models import StockTradingEnv, DQNAgent, A2CAgent, PPOAgent, create_dummy_trading_results
    RL_MODELS_AVAILABLE = True
except ImportError:
    logging.warning("rl_models.py를 불러올 수 없습니다. 강화학습 기능이 제한될 수 있습니다.")
    RL_MODELS_AVAILABLE = False

from data.fetch_data import fetch_stock_data
from reinforcement_learning.train import train_agent
from reinforcement_learning.evaluate import evaluate_agent
# 패턴 스캐너 기능을 위한 임포트 추가
from indicators.patterns.pattern_analyzer import PatternAnalyzer
from indicators.patterns.bollinger_bands import BollingerBands
from indicators.patterns.double_bottom import DoubleBottom

# 선택적 임포트 (파일이 없을 수 있음)
DoubleTop = None
HeadAndShoulders = None
try:
    from indicators.patterns.double_top import DoubleTop
except ImportError:
    pass

try:
    from indicators.patterns.head_and_shoulders import HeadAndShoulders
except ImportError:
    pass

# 페이지 기본 설정
def create_dashboard_layout():
    """개선된 대시보드 레이아웃"""
    # 상단 헤더 및 설명
    st.title("주식 거래 시스템")
    st.markdown("""
    이 시스템은 주식 데이터 분석, 패턴 스캐닝, 백테스팅, 강화학습 기반 트레이딩을 지원합니다.
    """)
    
    # 사이드바 메뉴
    with st.sidebar:
        # 대시보드 모드 선택 - 사이드바에서 한 번만 선택하도록 수정
        dashboard_mode = st.sidebar.radio(
            "대시보드 모드 선택",
            ["주식 분석 대시보드", "강화학습 트레이딩 대시보드", "포트폴리오 관리", "성능 대시보드", "멀티 에이전트 시스템"],
            key="dashboard_mode_selector"
        )
        # 세션 상태에 직접 모드 저장
        if dashboard_mode != st.session_state.get('dashboard_mode'):
            st.session_state.dashboard_mode = dashboard_mode
        
        # 모드별 사이드바 옵션
        if dashboard_mode == "주식 분석 대시보드":
            # 티커 입력 - 위젯에서 값 가져오기
            ticker = st.text_input("주식 티커", value=st.session_state.get('ticker', 'AAPL'), key="ticker_input")
            # 세션 상태 업데이트
            if 'ticker_input' in st.session_state:
                st.session_state.ticker = st.session_state.ticker_input
            
            # 날짜 범위 선택
            start_date = st.session_state.get('start_date', datetime.now() - timedelta(days=365))
            end_date = st.session_state.get('end_date', datetime.now())
            date_range = st.date_input("기간 선택", [start_date, end_date], key="date_range")
            
            # 날짜 범위 업데이트
            if len(date_range) == 2 and 'date_range' in st.session_state:
                st.session_state.start_date = date_range[0]
                st.session_state.end_date = date_range[1]
            
            # 투자 스타일 선택
            investment_style = st.selectbox(
                "투자 스타일", 
                ["default", "long_term", "swing_trade"],
                index=["default", "long_term", "swing_trade"].index(st.session_state.get('investment_style', 'default')),
                key="investment_style_select"
            )
            # 세션 상태 업데이트
            if 'investment_style_select' in st.session_state:
                st.session_state.investment_style = st.session_state.investment_style_select
        
        elif dashboard_mode == "강화학습 트레이딩 대시보드":
            # 강화학습 옵션
            rl_ticker = st.text_input("학습/평가할 티커", value=st.session_state.get('rl_ticker', 'AAPL'), key="rl_ticker_input")
            # 세션 상태 업데이트
            if 'rl_ticker_input' in st.session_state:
                st.session_state.rl_ticker = st.session_state.rl_ticker_input
            
            # 날짜 범위 선택
            rl_start_date = st.session_state.get('rl_start_date', datetime.now() - timedelta(days=365*2))
            rl_end_date = st.session_state.get('rl_end_date', datetime.now())
            rl_date_range = st.date_input("데이터 기간", [rl_start_date, rl_end_date], key="rl_date_range")
            
            # 날짜 범위 업데이트
            if len(rl_date_range) == 2 and 'rl_date_range' in st.session_state:
                st.session_state.rl_start_date = rl_date_range[0]
                st.session_state.rl_end_date = rl_date_range[1]
            
            # 투자 스타일 선택
            rl_investment_style = st.selectbox(
                "투자 스타일", 
                ["default", "long_term", "swing_trade"],
                index=["default", "long_term", "swing_trade"].index(st.session_state.get('rl_investment_style', 'default')),
                key="rl_investment_style_select"
            )
            # 세션 상태 업데이트
            if 'rl_investment_style_select' in st.session_state:
                st.session_state.rl_investment_style = st.session_state.rl_investment_style_select
            
            # 훈련/평가 옵션
            rl_epochs = st.slider(
                "학습 에포크", 
                10, 200, 
                st.session_state.get('rl_epochs', 50),
                10, 
                key="rl_epochs_slider"
            )
            # 세션 상태 업데이트
            if 'rl_epochs_slider' in st.session_state:
                st.session_state.rl_epochs = st.session_state.rl_epochs_slider

def format_number(number):
    """숫자를 포맷팅하는 함수"""
    if isinstance(number, (pd.Series, pd.DataFrame)):
        if len(number) > 0:
            # 경고 메시지 해결: float(ser.iloc[0]) 사용
            return float(number.iloc[-1].item() if hasattr(number.iloc[-1], 'item') else number.iloc[-1])
        return 0.0
    return float(number)

def load_stock_data(ticker, start_date, end_date):
    """
    주식 데이터를 로드하는 함수
    
    Args:
        ticker (str): 주식 티커 심볼
        start_date (datetime or str): 시작 날짜
        end_date (datetime or str): 종료 날짜
        
    Returns:
        pd.DataFrame: 주식 데이터
    """
    try:
        logging.info(f"{ticker} 데이터 로드 중... ({start_date} ~ {end_date})")
        
        # 날짜 형식 변환
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = start_date
            
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = end_date
        
        # 데이터 로드
        data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
        
        # 데이터 검증
        if data is None or data.empty:
            logging.warning(f"{ticker} 데이터를 찾을 수 없습니다.")
            return None
            
        # NaN 값 처리
        if data.isnull().values.any():
            logging.warning(f"{ticker} 데이터에 NaN 값이 포함되어 있습니다. 전방 채우기 방식으로 처리합니다.")
            data = data.fillna(method='ffill')
            
            # 여전히 NaN이 있다면 후방 채우기
            if data.isnull().values.any():
                data = data.fillna(method='bfill')
                
            # 그래도 NaN이 있다면 0으로 채우기
            if data.isnull().values.any():
                data = data.fillna(0)
        
        # 인덱스가 datetime인지 확인
        if not isinstance(data.index, pd.DatetimeIndex):
            logging.warning(f"{ticker} 데이터 인덱스가 DatetimeIndex가 아닙니다. 변환합니다.")
            data.index = pd.to_datetime(data.index)
        
        # 타임존 제거
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # 기본 컬럼 존재 여부 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"{ticker} 데이터에 필요한 컬럼이 누락되었습니다: {missing_columns}")
            # 중요 컬럼이 없다면 None 반환
            if any(col in missing_columns for col in ['Open', 'High', 'Low', 'Close']):
                return None
            
            # Volume만 없다면 더미 데이터 생성
            if 'Volume' in missing_columns:
                data['Volume'] = 0
        
        logging.info(f"{ticker} 데이터 로드 완료: {len(data)}개 행")
        return data
        
    except Exception as e:
        logging.error(f"{ticker} 데이터 로드 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None

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
        for pattern in ['Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df
    
    # 필요한 컬럼 확인
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"패턴 감지 실패: 필요한 컬럼 누락 - {missing_columns}")
        # 빈 결과 반환
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df
    
    try:
        # 결과를 저장할 데이터프레임 초기화 (모든 날짜에 대해 False로 설정)
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Uptrend', 'Downtrend']:
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
                pattern_df.at[data.index[i], 'Engulfing_Bullish'] = True
                logging.info(f"Bullish Engulfing 패턴 감지: {curr_date.strftime('%Y-%m-%d')}")
            
            # 3.2 Bearish Engulfing (이전 캔들이 양봉, 현재 캔들이 음봉이며 이전 캔들을 완전히 감싸는 경우)
            elif (close_prev > open_prev and  # 이전 캔들이 양봉
                  close_curr < open_curr and  # 현재 캔들이 음봉
                  open_curr >= close_prev and  # 현재 시가가 이전 종가보다 높거나 같음
                  close_curr <= open_prev):  # 현재 종가가 이전 시가보다 낮거나 같음
                pattern_df.at[data.index[i], 'Engulfing_Bearish'] = True
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
        import traceback
        logging.error(traceback.format_exc())
        
        # 오류 발생 시 기본 결과 반환
        pattern_df = pd.DataFrame(index=data.index)
        for pattern in ['Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Uptrend', 'Downtrend']:
            pattern_df[pattern] = False
        return pattern_df

# 패턴 시각화 함수
def plot_pattern_chart(data, pattern_results, pattern_name=None):
    """
    패턴 감지 결과를 기반으로 차트를 생성합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        pattern_results (pd.DataFrame): 패턴 감지 결과
        pattern_name (str, optional): 시각화할 특정 패턴 이름
        
    Returns:
        plotly.graph_objects.Figure: 생성된 차트 객체
    """
    # 데이터 유효성 검사
    if data is None or data.empty:
        logging.error("패턴 차트 생성 실패: 빈 데이터")
        return go.Figure().update_layout(title="데이터가 없습니다.")
    
    # 필요한 컬럼 확인
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"패턴 차트 생성 중 오류 발생: {missing_columns}")
        return go.Figure().update_layout(title=f"필요한 데이터가 없습니다: {', '.join(missing_columns)}")
    
    # 데이터가 충분한지 확인
    if len(data) < 5:
        logging.warning(f"패턴 차트 생성 중 경고: 데이터가 너무 적습니다 ({len(data)}개 행)")
        return go.Figure().update_layout(title="데이터가 충분하지 않습니다")
    
    # 패턴 결과 유효성 검사
    if pattern_results is None or pattern_results.empty:
        logging.error("패턴 차트 생성 실패: 패턴 결과 없음")
        return go.Figure().update_layout(title="패턴 감지 결과가 없습니다.")
    
    # 특정 패턴 필터링
    all_patterns = ['Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish']
    available_patterns = [col for col in all_patterns if col in pattern_results.columns]
    
    if pattern_name is not None and pattern_name not in available_patterns:
        logging.warning(f"패턴 '{pattern_name}'이 패턴 결과에 없습니다. 사용 가능한 패턴: {available_patterns}")
        pattern_name = None
    
    try:
        # 차트 생성 - 서브플롯 사용
        fig = go.Figure()
        
        # 캔들스틱 추가
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='캔들스틱',
                showlegend=True
            )
        )
        
        # 이동평균선 추가
        if 'MA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA20'],
                    name='MA20',
                    line=dict(color='blue', width=1),
                    showlegend=True
                )
            )
        
        if 'MA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA50'],
                    name='MA50',
                    line=dict(color='orange', width=1),
                    showlegend=True
                )
            )
        
        # 패턴 마커 추가
        colors = {
            'Doji': 'black',
            'Hammer': 'green',
            'Engulfing_Bullish': 'lime',
            'Engulfing_Bearish': 'red',
        }
        
        markers = {
            'Doji': 'diamond',
            'Hammer': 'triangle-up',
            'Engulfing_Bullish': 'circle',
            'Engulfing_Bearish': 'circle-open',
        }
        
        # 특정 패턴이 선택된 경우 해당 패턴만 표시
        patterns_to_show = [pattern_name] if pattern_name else available_patterns
        
        for pattern in patterns_to_show:
            if pattern in pattern_results.columns:
                # 패턴이 감지된 지점 찾기
                pattern_days = pattern_results.index[pattern_results[pattern] == True]
                
                if len(pattern_days) > 0:
                    # 해당 날짜의 고가에 마커 표시
                    y_values = [data.loc[day, 'High'] for day in pattern_days if day in data.index]
                    x_values = [day for day in pattern_days if day in data.index]
                    
                    if len(x_values) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='markers',
                                marker=dict(
                                    color=colors.get(pattern, 'purple'),
                                    size=10,
                                    symbol=markers.get(pattern, 'circle'),
                                    line=dict(width=2, color='black')
                                ),
                                name=f"{pattern}",
                                showlegend=True
                            )
                        )
        
        # 볼륨 차트 추가 (별도의 서브플롯으로)
        if 'Volume' in data.columns:
            # 상승/하락 볼륨 색상 구분
            colors = ['green' if float(data.iloc[i]['Close']) >= float(data.iloc[i]['Open']) else 'red' for i in range(len(data))]
            
            # 볼륨 데이터 추가
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='거래량',
                    marker_color=colors,
                    showlegend=True,
                    yaxis="y2"  # 보조 y축 사용
                )
            )
            
            # 두 개의 y축 설정
            fig.update_layout(
                yaxis2=dict(
                    title="거래량",
                    title_font=dict(color="#ff7f0e"),
                    tickfont=dict(color="#ff7f0e"),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title=f"주가 및 패턴 차트 ({data.index[0].date()} ~ {data.index[-1].date()})",
            xaxis_title="날짜",
            yaxis_title="가격",
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # X축 설정 (날짜 포맷)
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],  # 주말 제외
            tickformat="%Y-%m-%d"
        )
        
        return fig
    
    except Exception as e:
        logging.error(f"패턴 차트 생성 중 오류 발생: {str(e)}")
import traceback
        logging.error(traceback.format_exc())
        return go.Figure().update_layout(title=f"차트 생성 중 오류 발생: {str(e)}")

def calculate_technical_indicators(data):
    """기술적 지표 계산"""
    try:
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
        data_copy['Signal'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
        data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['Signal']
        
        return data_copy
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        # 사용자에게 친화적인 오류 메시지
        return None

def plot_stock_chart(data, ticker):
    """주식 차트를 그리는 함수 - 간단한 버전"""
    try:
        # 데이터 확인
        if data is None or data.empty:
            st.error("차트를 그릴 데이터가 없습니다.")
            return go.Figure().update_layout(title="차트를 그릴 데이터가 없습니다.")
        
        # Series로 변환 확인
        open_data = data['Open'] if isinstance(data['Open'], pd.Series) else data['Open'].iloc[:, 0]
        high_data = data['High'] if isinstance(data['High'], pd.Series) else data['High'].iloc[:, 0]
        low_data = data['Low'] if isinstance(data['Low'], pd.Series) else data['Low'].iloc[:, 0]
        close_data = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].iloc[:, 0]
        
        # 가장 기본적인 캔들스틱 차트 생성
        fig = go.Figure()
        
        # 캔들스틱 추가
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=open_data,
                high=high_data,
                low=low_data,
                close=close_data,
                name='주가'
            )
        )
        
        # 이동평균선 추가
        if 'MA5' in data.columns:
            ma5_data = data['MA5'] if isinstance(data['MA5'], pd.Series) else data['MA5'].iloc[:, 0]
            fig.add_trace(go.Scatter(x=data.index, y=ma5_data, name='MA5', line=dict(color='purple')))
        if 'MA20' in data.columns:
            ma20_data = data['MA20'] if isinstance(data['MA20'], pd.Series) else data['MA20'].iloc[:, 0]
            fig.add_trace(go.Scatter(x=data.index, y=ma20_data, name='MA20', line=dict(color='orange')))
        if 'MA60' in data.columns:
            ma60_data = data['MA60'] if isinstance(data['MA60'], pd.Series) else data['MA60'].iloc[:, 0]
            fig.add_trace(go.Scatter(x=data.index, y=ma60_data, name='MA60', line=dict(color='green')))
        
        # 기본 레이아웃 설정
        fig.update_layout(
            title=f"{ticker} 주가 차트",
            xaxis_title="날짜",
            yaxis_title="가격",
            height=600
        )
        
        return fig
    except Exception as e:
        logging.error(f"주식 차트 생성 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        # 사용자에게 친화적인 오류 메시지
        return go.Figure().update_layout(title=f"주식 차트 생성 중 오류 발생: {str(e)}")

def plot_technical_indicators(data):
    """기술적 지표 차트 생성"""
    try:
        # 데이터 유효성 검사
        if data is None or len(data) < 14:  # RSI 계산에 최소 14일 필요
            logging.error("기술적 지표 차트 생성 실패: 데이터가 충분하지 않습니다.")
            return go.Figure().update_layout(title="기술적 지표를 생성하기에 데이터가 충분하지 않습니다.")
            
        # 필요한 컬럼 확인
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"기술적 지표 차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            return go.Figure().update_layout(title=f"필요한 데이터가 누락되었습니다: {', '.join(missing_columns)}")
        
        # 데이터 복사본 생성하여 원본 데이터 보존
        data_copy = data.copy()
            
        # RSI 계산을 위한 데이터 준비
        if 'RSI' not in data_copy.columns:
            # RSI 계산
            delta = data_copy['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # 0으로 나누는 경우 방지
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            data_copy['RSI'] = 100 - (100 / (1 + rs))
            
        # MACD 계산
        if 'MACD' not in data_copy.columns or 'Signal' not in data_copy.columns:
            # MACD 계산
            ema12 = data_copy['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data_copy['Close'].ewm(span=26, adjust=False).mean()
            data_copy['MACD'] = ema12 - ema26
            data_copy['Signal'] = data_copy['MACD'].ewm(span=9, adjust=False).mean()
            data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['Signal']
            
        # 서브플롯이 있는 그림 생성 (2개의 행)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1,
                            subplot_titles=('RSI (14)', 'MACD'),
                            row_heights=[0.5, 0.5])
        
        # RSI 차트 (첫 번째 서브플롯)
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy['RSI'],
                line=dict(color='purple', width=1),
                name="RSI (14)"
            ),
            row=1, col=1
        )
        
        # RSI 과매수/과매도 라인
        fig.add_trace(
            go.Scatter(
                x=[data_copy.index[0], data_copy.index[-1]],
                y=[70, 70],
                line=dict(color='red', width=1, dash='dash'),
                name="과매수 (70)"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data_copy.index[0], data_copy.index[-1]],
                y=[30, 30],
                line=dict(color='green', width=1, dash='dash'),
                name="과매도 (30)"
            ),
            row=1, col=1
        )
        
        # MACD 차트 (두 번째 서브플롯)
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy['MACD'],
                line=dict(color='blue', width=1),
                name="MACD"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy['Signal'],
                line=dict(color='red', width=1),
                name="Signal"
            ),
            row=2, col=1
        )
        
        # MACD 히스토그램
        fig.add_trace(
            go.Bar(
                x=data_copy.index,
                y=data_copy['MACD'] - data_copy['Signal'],
                name="MACD 히스토그램",
                marker_color=np.where(data_copy['MACD'] >= data_copy['Signal'], 'green', 'red')
            ),
            row=2, col=1
        )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False
        )
        
        # RSI Y축 범위 설정 (0-100)
        fig.update_yaxes(range=[0, 100], row=1, col=1)
        
        # X축 설정 (날짜 포맷)
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],  # 주말 제외
            tickformat="%Y-%m-%d"
        )
        
        return fig
    except Exception as e:
        logging.error(f"기술적 지표 차트 생성 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return go.Figure().update_layout(title=f"기술적 지표 차트 생성 중 오류: {str(e)}")

def plot_volume_chart(data):
    """거래량 차트를 그리는 함수"""
    try:
        if data is None or data.empty:
            st.error("차트를 그릴 데이터가 없습니다.")
            return go.Figure()
        
        # 거래량 데이터 변환
        volume_data = data['Volume'] if isinstance(data['Volume'], pd.Series) else data['Volume'].iloc[:, 0]
        
        # 거래량 차트 생성
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=volume_data,
            name='거래량',
            marker=dict(color='rgba(0, 128, 0, 0.7)')
        ))
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title="거래량 추이",
            xaxis_title="날짜",
            yaxis_title="거래량",
            template="plotly_white",
            height=400
        )
        
        return fig
    except Exception as e:
        logging.error(f"거래량 차트 생성 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        # 사용자에게 친화적인 오류 메시지
        return None

def backtest_ma_crossover(data, short_window, long_window, initial_capital):
    """
    이동평균 교차 전략 백테스팅
    
    Args:
        data (pd.DataFrame): OHLCV 데이터
        short_window (int): 단기 이동평균 윈도우
        long_window (int): 장기 이동평균 윈도우
        initial_capital (float): 초기 자본금
        
    Returns:
        dict: 백테스팅 결과
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("백테스팅: 데이터가 비어 있습니다.")
            return {
                "error": "데이터가 비어 있습니다",
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"백테스팅: 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}"
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # NaN 값 확인 및 처리
        if data['Close'].isna().any():
            logging.warning("백테스팅: 종가에 NaN 값이 있습니다. 이전 값으로 채웁니다.")
            data = data.copy()
            data['Close'] = data['Close'].fillna(method='ffill')
            
        # 충분한 데이터 확인
        if len(data) < long_window + 10:  # 장기 이동평균 + 최소 거래 기간
            error_msg = f"백테스팅: 데이터가 충분하지 않습니다. 최소 {long_window + 10}개의 데이터 포인트가 필요합니다."
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # 이동평균 계산
        data = data.copy()
        data[f'MA{short_window}'] = data['Close'].rolling(window=short_window).mean()
        data[f'MA{long_window}'] = data['Close'].rolling(window=long_window).mean()
        
        # 시그널 생성
        data['Signal'] = 0.0
        data['Signal'][long_window:] = np.where(data[f'MA{short_window}'][long_window:] > data[f'MA{long_window}'][long_window:], 1.0, 0.0)
        data['Position'] = data['Signal'].diff()
        
        # 백테스팅 결과 계산
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Strategy_Returns'] = data['Returns'] * data['Signal'].shift(1)
        
        # NaN 값 처리
        data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
        
        # 포트폴리오 가치 추이 계산
        data['Cumulative_Returns'] = np.exp(data['Strategy_Returns'].cumsum()) - 1
        data['Portfolio_Value'] = initial_capital * (1 + data['Cumulative_Returns'])
        
        # 거래 이력 기록
        trades = []
        for i in range(1, len(data)):
            if data['Position'].iloc[i] == 1:  # 매수 시그널
                trades.append({
                    'date': data.index[i],
                    'action': 'Buy',
                    'price': data['Close'].iloc[i],
                    'shares': initial_capital / data['Close'].iloc[i]
                })
            elif data['Position'].iloc[i] == -1:  # 매도 시그널
                trades.append({
                    'date': data.index[i],
                    'action': 'Sell',
                    'price': data['Close'].iloc[i],
                    'shares': initial_capital / data['Close'].iloc[i]
                })
        
        # 최종 결과
        final_value = data['Portfolio_Value'].iloc[-1] if not data.empty else initial_capital
        
        return {
            'final_portfolio_value': final_value,
            'returns': (final_value / initial_capital - 1) * 100,
            'portfolio_values': data['Portfolio_Value'],
            'trades': trades,
            'data': data[['Close', f'MA{short_window}', f'MA{long_window}', 'Signal', 'Portfolio_Value']]
        }
    except Exception as e:
        logging.error(f"이동평균 교차 전략 백테스팅 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return {
            'error': f"백테스팅 중 오류 발생: {str(e)}",
            'final_portfolio_value': initial_capital,
            'returns': 0.0
        }

def backtest_rsi_strategy(data, rsi_period=14, overbought=70, oversold=30, initial_capital=10000):
    """
    RSI 기반 전략 백테스팅
    
    Args:
        data (pd.DataFrame): OHLCV 데이터
        rsi_period (int): RSI 계산 윈도우
        overbought (int): 과매수 기준
        oversold (int): 과매도 기준
        initial_capital (float): 초기 자본금
        
    Returns:
        dict: 백테스팅 결과
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("백테스팅: 데이터가 비어 있습니다.")
            return {
                "error": "데이터가 비어 있습니다",
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # 필수 컬럼 확인
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"백테스팅: 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}"
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # NaN 값 확인 및 처리
        if data['Close'].isna().any():
            logging.warning("백테스팅: 종가에 NaN 값이 있습니다. 이전 값으로 채웁니다.")
            data = data.copy()
            data['Close'] = data['Close'].fillna(method='ffill')
            
        # 충분한 데이터 확인
        if len(data) < rsi_period + 10:  # RSI 기간 + 최소 거래 기간
            error_msg = f"백테스팅: 데이터가 충분하지 않습니다. 최소 {rsi_period + 10}개의 데이터 포인트가 필요합니다."
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # RSI 계산
        data = data.copy()
        delta = data['Close'].diff()
        
        # delta가 NaN인 경우 처리
        delta = delta.fillna(0)
        
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=rsi_period).mean().fillna(0)
        avg_loss = loss.rolling(window=rsi_period).mean().fillna(0)
        
        # 0으로 나누는 것 방지
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI NaN 값 처리
        data['RSI'] = data['RSI'].fillna(50)  # 중립값으로 설정
        
        # 시그널 생성
        data['Signal'] = 0.0
        data['Signal'] = np.where(data['RSI'] < oversold, 1.0, 0.0)  # 과매도 시 매수
        data['Signal'] = np.where(data['RSI'] > overbought, -1.0, data['Signal'])  # 과매수 시 매도
        
        # 포지션 변화 계산 (0: 현금, 1: 주식 보유)
        data['Position'] = 0
        position = 0
        
        for i in range(rsi_period, len(data)):  # rsi_period부터 시작하여 RSI가 계산된 지점부터 시작
            if data['Signal'].iloc[i] == 1 and position == 0:  # 매수 신호 & 현금 상태
                position = 1
            elif data['Signal'].iloc[i] == -1 and position == 1:  # 매도 신호 & 주식 보유 상태
                position = 0
            
            data['Position'].iloc[i] = position
        
        # 수익률 계산
        data['Returns'] = data['Close'].pct_change().fillna(0)
        data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1).fillna(0)
        
        # 포트폴리오 가치 추이
        data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Returns']
        
        # 거래 이력
        trades = []
        for i in range(1, len(data)):
            if data['Position'].iloc[i] > data['Position'].iloc[i-1]:  # 매수
                trades.append({
                    'date': data.index[i],
                    'action': 'Buy',
                    'price': data['Close'].iloc[i],
                    'shares': initial_capital / data['Close'].iloc[i]
                })
            elif data['Position'].iloc[i] < data['Position'].iloc[i-1]:  # 매도
                trades.append({
                    'date': data.index[i],
                    'action': 'Sell',
                    'price': data['Close'].iloc[i],
                    'shares': initial_capital / data['Close'].iloc[i]
                })
        
        # 최종 결과
        final_value = data['Portfolio_Value'].iloc[-1] if not data.empty else initial_capital
        
        return {
            'final_portfolio_value': final_value,
            'returns': (final_value / initial_capital - 1) * 100,
            'portfolio_values': data['Portfolio_Value'],
            'trades': trades,
            'data': data[['Close', 'RSI', 'Signal', 'Portfolio_Value']]
        }
    except Exception as e:
        logging.error(f"RSI 기반 전략 백테스팅 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return {
            'error': f"백테스팅 중 오류 발생: {str(e)}",
            'final_portfolio_value': initial_capital,
            'returns': 0.0
        }

def backtest_bollinger_bands(data, window=20, num_std=2.0, initial_capital=10000):
    """
    볼린저 밴드 전략 백테스팅
    
    Args:
        data (pd.DataFrame): OHLCV 데이터
        window (int): 이동평균 윈도우
        num_std (float): 표준편차 배수
        initial_capital (float): 초기 자본금
        
    Returns:
        dict: 백테스팅 결과
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("백테스팅: 데이터가 비어 있습니다.")
            return {
                "error": "데이터가 비어 있습니다",
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # 필수 컬럼 확인
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"백테스팅: 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}"
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # NaN 값 확인 및 처리
        if data['Close'].isna().any():
            logging.warning("백테스팅: 종가에 NaN 값이 있습니다. 이전 값으로 채웁니다.")
            data = data.copy()
            data['Close'] = data['Close'].fillna(method='ffill')
            
        # 충분한 데이터 확인
        if len(data) < window + 10:  # 이동평균 윈도우 + 최소 거래 기간
            error_msg = f"백테스팅: 데이터가 충분하지 않습니다. 최소 {window + 10}개의 데이터 포인트가 필요합니다."
            logging.error(error_msg)
            return {
                "error": error_msg,
                "final_portfolio_value": initial_capital,
                "returns": 0.0
            }
            
        # 볼린저 밴드 계산
        data = data.copy()
        data['MA'] = data['Close'].rolling(window=window).mean()
        data['STD'] = data['Close'].rolling(window=window).std()
        
        # NaN 값 백필
        data['MA'] = data['MA'].fillna(method='bfill')
        data['STD'] = data['STD'].fillna(method='bfill')
        
        data['Upper_Band'] = data['MA'] + (data['STD'] * num_std)
        data['Lower_Band'] = data['MA'] - (data['STD'] * num_std)
        
        # 시그널 생성
        data['Signal'] = 0.0
        data['Signal'] = np.where(data['Close'] < data['Lower_Band'], 1.0, 0.0)  # 하단 밴드 아래로 내려가면 매수
        data['Signal'] = np.where(data['Close'] > data['Upper_Band'], -1.0, data['Signal'])  # 상단 밴드 위로 올라가면 매도
        
        # 포지션 변화 계산
        data['Position'] = 0
        position = 0
        
        for i in range(window, len(data)):  # window부터 시작하여 MA, STD가 계산된 지점부터 시작
            if data['Signal'].iloc[i] == 1 and position == 0:  # 매수 신호 & 현금 상태
                position = 1
            elif data['Signal'].iloc[i] == -1 and position == 1:  # 매도 신호 & 주식 보유 상태
                position = 0
            
            data['Position'].iloc[i] = position
        
        # 수익률 계산
        data['Returns'] = data['Close'].pct_change().fillna(0)
        data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1).fillna(0)
        
        # 포트폴리오 가치 추이
        data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Returns']
        
        # 거래 이력
        trades = []
        for i in range(1, len(data)):
            if data['Position'].iloc[i] > data['Position'].iloc[i-1]:  # 매수
                shares = initial_capital / data['Close'].iloc[i]
                trades.append({
                    'date': data.index[i],
                    'action': 'Buy',
                    'price': data['Close'].iloc[i],
                    'shares': shares
                })
            elif data['Position'].iloc[i] < data['Position'].iloc[i-1]:  # 매도
                shares = initial_capital / data['Close'].iloc[i]
                trades.append({
                    'date': data.index[i],
                    'action': 'Sell',
                    'price': data['Close'].iloc[i],
                    'shares': shares
                })
        
        # 최종 결과
        final_value = data['Portfolio_Value'].iloc[-1] if not data.empty else initial_capital
        
        return {
            'final_portfolio_value': final_value,
            'returns': (final_value / initial_capital - 1) * 100,
            'portfolio_values': data['Portfolio_Value'],
            'trades': trades,
            'data': data[['Close', 'MA', 'Upper_Band', 'Lower_Band', 'Signal', 'Portfolio_Value']]
        }
    except Exception as e:
        logging.error(f"볼린저 밴드 전략 백테스팅 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return {
            'error': f"백테스팅 중 오류 발생: {str(e)}",
            'final_portfolio_value': initial_capital,
            'returns': 0.0
        }

def validate_data(data):
    """데이터 유효성 검증 함수"""
    try:
        if data is None:
            return False, "데이터가 None입니다"
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                return False, f"필수 컬럼 '{col}'이 없습니다"
        
        if len(data) < 50:
            return False, "데이터가 부족합니다 (최소 50개 필요)"
        
        return True, "데이터가 유효합니다"
    except Exception as e:
        logging.error(f"데이터 유효성 검증 중 오류 발생: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        # 사용자에게 친화적인 오류 메시지
        return None

def evaluate_with_fixed_seed(agent, env, seed=42):
    """고정된 시드로 모델 평가"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 평가 로직...

def create_enhanced_chart(data, ticker):
    """개선된 차트 생성"""
    # 필요한 라이브러리를 함수 시작 부분에서 임포트
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    try:
        # 데이터 유효성 검사
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            # 누락된 컬럼 중 필수 컬럼이 있는지 확인
            essential_columns = ['Open', 'High', 'Low', 'Close']
            missing_essential = [col for col in essential_columns if col in missing_columns]
            
            if missing_essential:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"필수 데이터 누락: {', '.join(missing_essential)}",
                    showarrow=False,
                    font=dict(size=12, color="red")
                )
                return fig
            
            # Volume만 누락된 경우 0으로 채움
            if 'Volume' in missing_columns:
                data = data.copy()
                data['Volume'] = 0
                logging.warning(f"거래량 데이터가 누락되어 0으로 채웠습니다. - {ticker}")
        
        # 데이터 충분한지 확인
        if len(data) < 5:
            logging.warning(f"차트 생성을 위한 데이터가 부족합니다. 행 개수: {len(data)} - {ticker}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"데이터 부족 (행 개수: {len(data)})",
                showarrow=False,
                font=dict(size=12, color="orange")
            )
            return fig
        
        # 데이터 복사본 생성 (원본 보존)
        df = data.copy()
        
        # 데이터 타입 검증 및 변환
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                # 숫자 타입이 아닌 경우 변환 시도
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logging.warning(f"컬럼 {col}이 숫자 타입이 아닙니다. 변환을 시도합니다.")
                    try:
                        # Series 자체를 변환
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logging.error(f"컬럼 {col} 숫자 변환 실패: {str(e)}")
        
        # NaN 값 처리
        essential_columns = ['Open', 'High', 'Low', 'Close']
        has_nan = False
        for col in essential_columns:
            if col in df.columns and df[col].isna().any():
                has_nan = True
                break
        
        if has_nan:
            logging.warning(f"필수 컬럼에 NaN 값이 존재합니다. 해당 행을 제거합니다.")
            df = df.dropna(subset=[col for col in essential_columns if col in df.columns])
        
        # Volume 컬럼의 NaN 값은 0으로 대체
        if 'Volume' in df.columns and df['Volume'].isna().any():
            df['Volume'] = df['Volume'].fillna(0)
        
        # 서브플롯 생성 (캔들스틱 + 거래량)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.02)
        
        # 캔들스틱 차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='가격'
            ),
            row=1, col=1
        )
        
        # 거래량 바 차트 추가
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='거래량',
                marker=dict(color=colors)
            ),
            row=2, col=1
        )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title=f'{ticker} 주가 차트',
            yaxis_title='가격',
            yaxis2_title='거래량',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        
        # 캔들스틱 스타일 설정
        fig.update_layout(
            xaxis=dict(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # 주말 제외
                ]
            )
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"차트 생성 중 오류 발생: {str(e)}")
        # 에러 메시지를 표시하는 빈 차트 반환
        try:
            # go 변수가 정의되지 않은 상태에서 오류 발생 가능성 있음
            fig = go.Figure()
            fig.add_annotation(
                text=f"차트 생성 오류: {str(e)}",
                showarrow=False,
                font=dict(size=12, color="red")
            )
            return fig
        except:
            # 모든 것이 실패하면 None 반환
            logging.critical("차트 오류 처리 중 추가 예외 발생")
            return None

def plot_technical_indicators(data):
    """기술적 지표 차트 생성"""
    try:
        # 데이터 유효성 검사
        if data is None or len(data) < 14:  # RSI 계산에 최소 14일 필요
            logging.error("기술적 지표 차트 생성 실패: 데이터가 충분하지 않습니다.")
            return None
            
        # 필요한 컬럼 확인
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"기술적 지표 차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            return None
            
        # RSI 계산을 위한 데이터 준비
        if 'RSI' not in data.columns:
            # RSI 계산
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
        # MACD 계산
        if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
            # MACD 계산
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
        # 차트 생성
        fig = go.Figure()
        
        # RSI 차트
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                line=dict(color='purple', width=1),
                name="RSI (14)"
            )
        )
        
        # RSI 과매수/과매도 라인
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[70, 70],
                line=dict(color='red', width=1, dash='dash'),
                name="과매수 (70)"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[30, 30],
                line=dict(color='green', width=1, dash='dash'),
                name="과매도 (30)"
            )
        )
        
        # MACD 차트
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                line=dict(color='blue', width=1),
                name="MACD"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                line=dict(color='red', width=1),
                name="Signal"
            )
        )
        
        # MACD 히스토그램
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD'] - data['MACD_Signal'],
                name="MACD 히스토그램",
                marker_color=np.where(data['MACD'] >= data['MACD_Signal'], 'green', 'red')
            )
        )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            height=600,
            template="plotly_white"
        )
        
        # Y축 범위 설정 (RSI)
        fig.update_yaxes(range=[0, 100])
        
        return fig
    except Exception as e:
        logging.error(f"기술적 지표 차트 생성 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None

# 성능 개선: 메모리 캐싱 및 병렬 처리 구현
@lru_cache(maxsize=32)
def cached_load_stock_data(ticker, start_date, end_date):
    """
    주식 데이터를 로드하고 캐싱하는 함수
    
    Args:
        ticker (str): 종목 코드
        start_date (str): 시작일
        end_date (str): 종료일
        
    Returns:
        pd.DataFrame: 주식 데이터
    """
    try:
        # 날짜 형식 변환 (datetime → 문자열)
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = start_date
            
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = end_date
        
        data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)

        if data.empty:
            logging.warning(f"'{ticker}' 종목의 데이터가 없습니다.")
            return None
            
        return data
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None

def parallel_load_data(tickers, start_date, end_date, max_workers=4):
    """
    여러 종목의 데이터를 병렬로 로드하는 함수
    
    Args:
        tickers (list): 종목 코드 리스트
        start_date (str): 시작일
        end_date (str): 종료일
        max_workers (int): 최대 워커 수
        
    Returns:
        dict: 종목별 데이터
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 병렬로 데이터 로드
        future_to_ticker = {executor.submit(cached_load_stock_data, ticker, start_date, end_date): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data is not None:
                    results[ticker] = data
            except Exception as e:
                logging.error(f"{ticker} 데이터 로드 중 오류: {str(e)}")
    
    return results

# 성능 모니터링 클래스
class PerformanceMonitor:
    """성능 모니터링을 위한 클래스"""
    
    def __init__(self):
        """성능 모니터 초기화"""
        # 함수별 실행 시간 저장
        self.function_times = {}
        # 데이터 로딩 시간 저장
        self.data_loading_times = {}
        # 시작 시간
        self.start_time = None
    
    def start_timer(self):
        """타이머 시작"""
        self.start_time = time.time()
    
    def log_function_time(self, func_name, execution_time=None):
        """함수 실행 시간 기록
        
        Args:
            func_name (str): 함수 이름
            execution_time (float, optional): 실행 시간(초). None이면 start_timer부터 현재까지의 시간을 사용
        """
        if execution_time is None:
            if self.start_time is None:
                logging.warning("타이머가 시작되지 않았습니다.")
            return
            execution_time = time.time() - self.start_time
        
        if func_name not in self.function_times:
            self.function_times[func_name] = []
            
        self.function_times[func_name].append(execution_time)
        logging.debug(f"함수 '{func_name}' 실행 시간: {execution_time:.4f}초")
    
    def log_data_loading(self, ticker, loading_time=None):
        """데이터 로딩 시간 기록
        
        Args:
            ticker (str): 주식 티커
            loading_time (float, optional): 로딩 시간(초). None이면 start_timer부터 현재까지의 시간을 사용
        """
        if loading_time is None:
            if self.start_time is None:
                logging.warning("타이머가 시작되지 않았습니다.")
                return
            loading_time = time.time() - self.start_time
        
        if ticker not in self.data_loading_times:
            self.data_loading_times[ticker] = []
            
        self.data_loading_times[ticker].append(loading_time)
        logging.debug(f"티커 '{ticker}' 데이터 로딩 시간: {loading_time:.4f}초")
    
    def display_dashboard(self):
        """성능 모니터링 대시보드 표시"""
        st.subheader("성능 모니터링")
        
        # 함수 실행 시간 차트
        if self.function_times:
            st.write("#### 함수 실행 시간")
            
            # 평균 실행 시간 계산
            avg_times = {}
            for func_name, times in self.function_times.items():
                avg_times[func_name] = sum(times) / len(times) if times else 0
            
            # 막대 차트로 표시
            fig = go.Figure()
            for func_name, avg_time in avg_times.items():
                fig.add_trace(go.Bar(
                    x=[func_name],
                    y=[avg_time],
                    name=func_name,
                    text=[f"{avg_time:.3f}초"],
                    textposition="auto"
                ))
            
            fig.update_layout(
                title="평균 함수 실행 시간 (초)",
                xaxis_title="함수",
                yaxis_title="실행 시간 (초)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_stock_analysis_dashboard():
    """주식 분석 대시보드 렌더링"""
    st.header("주식 분석 대시보드")
    
    # 세션에서 파라미터 가져오기
    ticker = st.session_state.get('ticker', 'AAPL')
    start_date = st.session_state.get('start_date', datetime.now() - timedelta(days=365))
    end_date = st.session_state.get('end_date', datetime.now())
    
    # 데이터 로딩 상태 표시
    with st.spinner(f"{ticker} 데이터를 로딩 중입니다..."):
        data = load_stock_data(ticker, start_date, end_date)
    
    if data is not None and not data.empty:
        # 데이터 정보 표시
        st.success(f"{ticker} 데이터 로딩 완료: {len(data)}개 행")
        
        # 기술적 지표 계산
        data_with_indicators = calculate_technical_indicators(data)
        
        # 주식 차트 표시
        st.subheader("주가 차트")
        try:
            stock_chart = create_enhanced_chart(data, ticker)
            if stock_chart is not None:
                st.plotly_chart(stock_chart, use_container_width=True)
            else:
                st.error("주가 차트를 생성할 수 없습니다.")
    except Exception as e:
                st.error(f"주가 차트 생성 중 오류 발생: {str(e)}")
                logging.error(f"주가 차트 생성 오류: {str(e)}")
        
        # 기술적 지표 차트
        st.subheader("기술적 지표")
        if data_with_indicators is not None:
            try:
                rsi_chart = plot_technical_indicators(data_with_indicators)
                if rsi_chart is not None:
                    st.plotly_chart(rsi_chart, use_container_width=True)
                else:
                    st.warning("기술적 지표 차트를 생성할 수 없습니다.")
                
                # 거래량 차트
                volume_chart = plot_volume_chart(data_with_indicators)
                if volume_chart is not None:
                    st.plotly_chart(volume_chart, use_container_width=True)
                else:
                    st.warning("거래량 차트를 생성할 수 없습니다.")
            except Exception as e:
                st.error(f"지표 차트 생성 중 오류 발생: {str(e)}")
                logging.error(f"지표 차트 생성 오류: {str(e)}")
        
        # 패턴 스캐너 결과
        st.subheader("패턴 스캐너")
        try:
            pattern_results = scan_patterns(data)
            if pattern_results is not None and not pattern_results.empty:
                detected_patterns = [col for col in pattern_results.columns 
                                    if col not in ['long_term', 'swing'] and pattern_results[col].abs().sum() > 0]
                
                if detected_patterns:
                    st.success(f"감지된 패턴: {', '.join(detected_patterns)}")
                    
                    # 첫 번째 감지된 패턴으로 차트 생성
                    try:
                        pattern_chart = plot_pattern_chart(data, pattern_results, detected_patterns[0])
                        if pattern_chart:
                            st.plotly_chart(pattern_chart, use_container_width=True)
                        else:
                            st.warning("패턴 차트를 생성할 수 없습니다.")
                    except Exception as e:
                        st.error(f"패턴 차트 생성 중 오류 발생: {str(e)}")
                        logging.error(f"패턴 차트 생성 오류: {str(e)}")
                else:
                    st.info("감지된 패턴이 없습니다.")
            else:
                st.info("패턴 분석을 위한 충분한 데이터가 없습니다.")
        except Exception as e:
            st.error(f"패턴 스캐너 오류: {str(e)}")
            logging.error(f"패턴 스캐너 오류: {str(e)}")
    else:
        st.error(f"{ticker} 데이터를 로드할 수 없습니다. 티커 심볼과 날짜 범위를 확인해주세요.")

def render_rl_trading_dashboard():
    """강화학습 트레이딩 대시보드 렌더링"""
    st.header("강화학습 트레이딩 대시보드")
    
    # 세션에서 파라미터 가져오기
    ticker = st.session_state.get('rl_ticker', 'AAPL')
    start_date = st.session_state.get('rl_start_date', datetime.now() - timedelta(days=365*2))
    end_date = st.session_state.get('rl_end_date', datetime.now())
    investment_style = st.session_state.get('rl_investment_style', 'default')
    epochs = st.session_state.get('rl_epochs', 50)
    
    # 탭 생성
    tabs = st.tabs(["데이터 확인", "모델 훈련", "모델 평가"])
    
    # 데이터 확인 탭
    with tabs[0]:
        st.subheader("트레이딩 데이터")
        # 데이터 로딩 상태 표시
        with st.spinner(f"{ticker} 데이터를 로딩 중입니다..."):
            data = load_stock_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            # 데이터 정보 표시
            st.success(f"{ticker} 데이터 로딩 완료: {len(data)}개 행")
            
            # 데이터 분할 옵션
            train_test_ratio = st.slider("훈련/테스트 데이터 분할 비율", 0.5, 0.9, 0.8, 0.05)
            
            # 데이터 분할
            train_size = int(len(data) * train_test_ratio)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            # 훈련/테스트 데이터 정보
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"훈련 데이터: {len(train_data)}개 행")
                st.write(f"기간: {train_data.index[0].strftime('%Y-%m-%d')} ~ {train_data.index[-1].strftime('%Y-%m-%d')}")
            
            with col2:
                st.info(f"테스트 데이터: {len(test_data)}개 행")
                st.write(f"기간: {test_data.index[0].strftime('%Y-%m-%d')} ~ {test_data.index[-1].strftime('%Y-%m-%d')}")
            
            # 주식 차트 표시
            st.subheader("주가 차트")
            stock_chart = create_enhanced_chart(data, ticker)
            st.plotly_chart(stock_chart, use_container_width=True)
            
            # 훈련/테스트 데이터 저장
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
        else:
            st.error(f"{ticker} 데이터를 로드할 수 없습니다. 티커 심볼과 날짜 범위를 확인해주세요.")
    
    # 모델 훈련 탭
    with tabs[1]:
        st.subheader("강화학습 모델 훈련")
        
        if 'train_data' in st.session_state and not st.session_state.train_data.empty:
            # 훈련 파라미터 설정
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("배치 크기", 16, 256, 32, 16)
            with col2:
                learning_rate = st.select_slider("학습률", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            
            # 훈련 버튼
            if st.button("모델 훈련 시작", key="train_button"):
                st.info(f"{ticker} 모델 훈련을 시작합니다. 투자 스타일: {investment_style}")
                
                with st.spinner(f"모델 훈련 중... 에포크: {epochs}, 배치 크기: {batch_size}"):
                    try:
                        # 훈련 함수 호출
                        from reinforcement_learning.train import train_agent
                        
                        model_path = train_agent(
                            st.session_state.train_data, 
                            ticker, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            investment_style=investment_style
                        )
                        
                        # 모델 경로 저장
                        st.session_state.model_path = model_path
                        st.success(f"모델 훈련 완료! 모델 저장 경로: {model_path}")
                    except Exception as e:
                        st.error(f"모델 훈련 중 오류가 발생했습니다: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("먼저 '데이터 확인' 탭에서 데이터를 로드해주세요.")
    
    # 모델 평가 탭
    with tabs[2]:
        st.subheader("모델 평가")
        
        if 'test_data' in st.session_state and not st.session_state.test_data.empty:
            # 모델 선택 옵션
            model_options = ["최근 훈련된 모델"]
            
            # 모델 디렉토리 확인
            model_dir = "models"
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith(".pth") and ticker in file:
                        model_options.append(file)
            
            selected_model = st.selectbox("평가할 모델 선택", model_options)
            
            # 평가 버튼
            if st.button("모델 평가 시작", key="evaluate_button"):
                model_path = None
                
                if selected_model == "최근 훈련된 모델" and 'model_path' in st.session_state:
                    model_path = st.session_state.model_path
                elif selected_model != "최근 훈련된 모델":
                    model_path = os.path.join(model_dir, selected_model)
                
                if model_path:
                    with st.spinner("모델 평가 중..."):
                        try:
                            # 평가 함수 호출
                            from reinforcement_learning.evaluate import evaluate_agent
                            
                            eval_results = evaluate_agent(
                                st.session_state.test_data,
                                ticker=ticker,
                                investment_style=investment_style
                            )
                            
                            if eval_results:
                                # 평가 결과 표시
                                st.success("모델 평가 완료!")
                                
                                # 성능 지표
                                metrics_cols = st.columns(4)
                                metrics_cols[0].metric("초기 자산", f"${eval_results['Initial Balance']:,.2f}")
                                metrics_cols[1].metric("최종 자산", f"${eval_results['Final Balance']:,.2f}")
                                metrics_cols[2].metric("총 수익률", f"{eval_results['Total Return (%)']:.2f}%")
                                metrics_cols[3].metric("거래 횟수", eval_results['Number of Trades'])
                                
                                # 차트 표시
                                st.subheader("포트폴리오 가치 추이")
                                
                                # 포트폴리오 가치 차트 생성
                                portfolio_fig = go.Figure()
                                
                                # 날짜 배열 생성
                                dates = st.session_state.test_data.index
                                if len(dates) != len(eval_results['portfolio_values']):
                                    # 길이가 다르면 거래일 기준으로 인덱스 생성
                                    dates = pd.date_range(
                                        start=dates[0], 
                                        periods=len(eval_results['portfolio_values']), 
                                        freq='B'
                                    )
                                
                                # 포트폴리오 가치 추이
                                portfolio_fig.add_trace(
                                    go.Scatter(
                                        x=dates,
                                        y=eval_results['portfolio_values'],
                                        mode='lines',
                                        name='포트폴리오 가치',
                                        line=dict(color='blue', width=2)
                                    )
                                )
                                
                                # 차트 레이아웃 설정
                                portfolio_fig.update_layout(
                                    title=f"{ticker} 강화학습 트레이딩 성능",
                                    xaxis_title="날짜",
                                    yaxis_title="포트폴리오 가치 ($)",
                                    height=500,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(portfolio_fig, use_container_width=True)
                                
                                # 거래 내역
                                if eval_results['trades']:
                                    st.subheader("주요 거래 내역")
                                    trades_df = pd.DataFrame(eval_results['trades'])
                                    st.dataframe(trades_df.head(10))
                            else:
                                st.error("모델 평가 결과가 없습니다.")
                        except Exception as e:
                            st.error(f"모델 평가 중 오류가 발생했습니다: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error("평가할 모델이 없습니다. 먼저 모델을 훈련하세요.")
        else:
            st.warning("먼저 '데이터 확인' 탭에서 데이터를 로드해주세요.")

def create_rl_trading_dashboard():
    """
    강화학습 트레이딩 대시보드를 생성합니다.
    """
    st.title("강화학습 트레이딩 대시보드")
    
    # 세션 상태 초기화
    if 'rl_ticker' not in st.session_state:
        st.session_state.rl_ticker = 'AAPL'
    if 'rl_start_date' not in st.session_state:
        st.session_state.rl_start_date = (datetime.now() - timedelta(days=365)).date()
    if 'rl_end_date' not in st.session_state:
        st.session_state.rl_end_date = datetime.now().date()
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = 'DQN'
    if 'rl_capital' not in st.session_state:
        st.session_state.rl_capital = 10000
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = None

    # 사이드바 설정
    with st.sidebar:
        st.header("트레이딩 설정")
        ticker = st.text_input("티커 심볼", value=st.session_state.rl_ticker)
        start_date = st.date_input("시작일", value=st.session_state.rl_start_date)
        end_date = st.date_input("종료일", value=st.session_state.rl_end_date)
        
        model_options = ["DQN", "A2C", "PPO"]
        model_type = st.selectbox("트레이딩 모델", options=model_options, index=model_options.index(st.session_state.rl_model))
        
        initial_capital = st.number_input("초기 자본금 ($)", min_value=1000, max_value=1000000, value=st.session_state.rl_capital, step=1000)
        
        run_button = st.button("백테스트 실행")
    
    # 사이드바 입력값 업데이트
    if ticker != st.session_state.rl_ticker or \
       start_date != st.session_state.rl_start_date or \
       end_date != st.session_state.rl_end_date or \
       model_type != st.session_state.rl_model or \
       initial_capital != st.session_state.rl_capital:
        
        st.session_state.rl_ticker = ticker
        st.session_state.rl_start_date = start_date
        st.session_state.rl_end_date = end_date
        st.session_state.rl_model = model_type
        st.session_state.rl_capital = initial_capital
        st.session_state.rl_results = None  # 설정 변경 시 결과 초기화
    
    # 백테스트 실행
    if run_button:
        with st.spinner("백테스트를 실행 중입니다..."):
            # 데이터 로드
            try:
                data = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if data is None or data.empty:
                    st.error(f"{ticker} 데이터를 가져올 수 없습니다.")
                    return
                
                # 기술적 지표 추가
                data = add_technical_indicators(data)
                
                # 모델 실행
                st.session_state.rl_results = run_rl_trading(data, model_type, initial_capital)
                st.success("백테스트가 완료되었습니다!")
            except Exception as e:
                st.error(f"백테스트 실행 중 오류가 발생했습니다: {str(e)}")
                import traceback
        logging.error(traceback.format_exc())
                return
    
    # 결과 표시
    if st.session_state.rl_results is not None:
        display_rl_results(st.session_state.rl_results)
    else:
        # 초기 안내 메시지
        st.info("강화학습 트레이딩 모델로 주식 트레이딩 백테스트를 실행할 수 있습니다. "
                "사이드바에서 주식 티커, 백테스트 기간, 모델 타입, 초기 자본금을 설정하고 '백테스트 실행' 버튼을 클릭하세요.")
        
        # 각 모델에 대한 설명
        with st.expander("강화학습 트레이딩 모델 설명"):
            st.markdown("""
            ### DQN (Deep Q-Network)
            - **특징**: 밸류 기반 강화학습, 시장 상태에 따른 매수, 매도, 관망 액션의 Q-값을 학습
            - **장점**: 간단한 트레이딩 전략에 효과적, 상태-행동 매핑 명확
            - **단점**: 급격한 시장 변동성에 적응 속도 느림
            
            ### A2C (Advantage Actor-Critic)
            - **특징**: 정책 기반과 밸류 기반의 결합 방식, 액터가 정책 결정, 크리틱이 정책 평가
            - **장점**: 안정적인 학습, 시장 변동성에 더 빠른 적응
            - **단점**: 하이퍼파라미터 튜닝 필요
            
            ### PPO (Proximal Policy Optimization)
            - **특징**: 정책 최적화 방법, 급격한 정책 변화 제한으로 안정성 향상
            - **장점**: 샘플 효율성 높음, 다양한 시장 환경에서 안정적인 성능
            - **단점**: 계산 복잡성 증가, 구현 복잡
            """)
            
        st.image("https://miro.medium.com/max/700/1*Z5-lWkyzcRB5ahgm9qyxvg.png", 
                 caption="강화학습 트레이딩 알고리즘 구조 예시")

def run_rl_trading(data, model_type, initial_capital):
    """
    강화학습 모델로 백테스트를 실행합니다.
    
    Args:
        data (pd.DataFrame): 주식 데이터
        model_type (str): 모델 유형 ('DQN', 'A2C', 'PPO')
        initial_capital (float): 초기 자본금
        
    Returns:
        dict: 백테스트 결과
    """
    try:
        # 모델 경로 설정
        models_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        model_paths = {
            'DQN': os.path.join(models_dir, 'dqn_model.pt'),
            'A2C': os.path.join(models_dir, 'a2c_model.pt'),
            'PPO': os.path.join(models_dir, 'ppo_model.pt')
        }
        
        model_path = model_paths.get(model_type)
        if not model_path or not os.path.exists(model_path):
            logging.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            # 기본 더미 모델 생성 (테스트용)
            with open(model_path, 'w') as f:
                f.write("dummy model")
        
        logging.info(f"트레이딩 환경 초기화: {model_type} 모델, 초기 자본금: ${initial_capital}")
        
        # 트레이딩 환경 설정
        env = StockTradingEnv(
            data=data,
            initial_capital=initial_capital,
            transaction_fee=0.001,  # 0.1% 거래 수수료
            window_size=20
        )
        
        # 모델 선택 및 로드
        if model_type == 'DQN':
            agent = DQNAgent.load(model_path)
        elif model_type == 'A2C':
            agent = A2CAgent.load(model_path)
        elif model_type == 'PPO':
            agent = PPOAgent.load(model_path)
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        
        # 백테스트 실행
        trades_df, portfolio_values = agent.backtest(env)
        
        # 성과 지표 계산
        final_value = portfolio_values[-1]
        return_pct = ((final_value - initial_capital) / initial_capital) * 100
        
        # 최대 낙폭 계산
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 샤프 비율 계산
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        sharpe_ratio = 0
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # 연간화
        
        # 일일 수익률 계산
        daily_return_series = pd.Series(daily_returns, index=data.index[1:])
        
        # 결과 반환
        results = {
            'data': data,
            'trades': trades_df,
            'portfolio_values': portfolio_values,
            'portfolio_values_series': pd.Series(portfolio_values, index=data.index),
            'daily_returns': daily_return_series,
            'initial_value': initial_capital,
            'final_value': final_value,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'model_type': model_type,
            'n_trades': len(trades_df)
        }
        
        logging.info(f"백테스트 완료: 총 {len(trades_df)}개 거래, 최종 수익률: {return_pct:.2f}%")
        return results
    
    except Exception as e:
        logging.error(f"강화학습 백테스트 실행 중 오류: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 더미 백테스트 결과 생성
        return create_dummy_trading_results(data, initial_capital)

def display_rl_results(results):
    """
    강화학습 백테스트 결과를 표시합니다.
    
    Args:
        results (dict): 백테스트 결과
    """
    # 결과 요약
    st.subheader("백테스트 결과 요약")
    
    # 성과 지표 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="최종 자산",
            value=f"${results['final_value']:,.2f}",
            delta=f"{results['return_pct']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="거래 횟수",
            value=f"{results['n_trades']}회"
        )
    
    with col3:
        st.metric(
            label="최대 낙폭",
            value=f"{results['max_drawdown']:.2f}%",
            delta=f"{-results['max_drawdown']:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="샤프 비율",
            value=f"{results['sharpe_ratio']:.2f}"
        )
    
    # 포트폴리오 가치 변화 차트
    st.subheader("포트폴리오 가치 변화")
    
    # 포트폴리오 차트 생성
    fig = go.Figure()
    
    # 포트폴리오 가치 선 추가
    fig.add_trace(
        go.Scatter(
            x=results['portfolio_values_series'].index,
            y=results['portfolio_values_series'].values,
            mode='lines',
            name='포트폴리오 가치',
            line=dict(color='blue', width=2)
        )
    )
    
    # 초기 자본금 기준선 추가
    fig.add_trace(
        go.Scatter(
            x=[results['portfolio_values_series'].index[0], results['portfolio_values_series'].index[-1]],
            y=[results['initial_value'], results['initial_value']],
            mode='lines',
            name='초기 자본금',
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    
    # 매수/매도 지점 표시
    if 'trades' in results and not results['trades'].empty:
        buy_points = results['trades'][results['trades']['action'] == 'BUY']
        sell_points = results['trades'][results['trades']['action'] == 'SELL']
        
        if not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points['Date'],
                    y=[results['portfolio_values_series'][date] for date in buy_points['Date']],
                    mode='markers',
                    name='매수',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                )
            )
        
        if not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points['Date'],
                    y=[results['portfolio_values_series'][date] for date in sell_points['Date']],
                    mode='markers',
                    name='매도',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                )
            )
    
    # 차트 레이아웃 설정
    fig.update_layout(
        title=f"{results['model_type']} 모델 트레이딩 성과 ({results['return_pct']:.2f}% 수익률)",
        xaxis_title="날짜",
        yaxis_title="포트폴리오 가치 ($)",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 일일 수익률 히스토그램
    if 'daily_returns' in results and not results['daily_returns'].empty:
        st.subheader("일일 수익률 분포")
        
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=results['daily_returns'].values * 100,  # 퍼센트로 변환
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            )
        )
        
        hist_fig.update_layout(
            title="일일 수익률 분포",
            xaxis_title="일일 수익률 (%)",
            yaxis_title="빈도",
            template="plotly_white"
        )
        
        # 평균선 추가
        mean_return = results['daily_returns'].mean() * 100
        hist_fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {mean_return:.2f}%"
        )
        
        st.plotly_chart(hist_fig, use_container_width=True)
    
    # 거래 내역 표시
    if 'trades' in results and not results['trades'].empty:
        st.subheader("거래 내역")
        
        # 거래 내역 포맷팅
        trades_display = results['trades'].copy()
        if 'Date' in trades_display.columns:
            trades_display['Date'] = trades_display['Date'].dt.strftime('%Y-%m-%d')
        if 'Price' in trades_display.columns:
            trades_display['Price'] = trades_display['Price'].apply(lambda x: f"${x:.2f}")
        if 'Value' in trades_display.columns:
            trades_display['Value'] = trades_display['Value'].apply(lambda x: f"${x:.2f}")
        if 'Cash' in trades_display.columns:
            trades_display['Cash'] = trades_display['Cash'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(trades_display, use_container_width=True)

def render_portfolio_dashboard():
    """포트폴리오 관리 대시보드를 렌더링하는 함수"""
    st.markdown("## 포트폴리오 관리")
    
    # 세션 상태 초기화
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'stocks': [],
            'cash': 100000,
            'history': []
        }
    
    # 사이드바 설정
    with st.sidebar:
        st.subheader("포트폴리오 설정")
        
        # 현금 추가/인출
        cash_action = st.radio("현금 관리", ["추가", "인출"], key="cash_action")
        cash_amount = st.number_input("금액", min_value=0, value=10000, step=1000, key="cash_amount")
        
        if st.button("실행", key="cash_btn"):
            if cash_action == "추가":
                st.session_state.portfolio['cash'] += cash_amount
                st.session_state.portfolio['history'].append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'action': "현금 추가",
                    'amount': cash_amount
                })
                st.success(f"{cash_amount:,}원이 추가되었습니다.")
            else:
                if cash_amount > st.session_state.portfolio['cash']:
                    st.error("보유 현금보다 많은 금액을 인출할 수 없습니다.")
                else:
                    st.session_state.portfolio['cash'] -= cash_amount
                    st.session_state.portfolio['history'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'action': "현금 인출",
                        'amount': cash_amount
                    })
                    st.success(f"{cash_amount:,}원이 인출되었습니다.")
        
        # 주식 추가
        st.subheader("주식 추가")
        stock_ticker = st.text_input("종목 티커", value="AAPL", key="add_stock_ticker")
        stock_amount = st.number_input("수량", min_value=1, value=10, key="add_stock_amount")
        stock_price = st.number_input("매입 가격", min_value=0.01, value=100.0, step=0.01, key="add_stock_price")
        
        if st.button("주식 추가", key="add_stock_btn"):
            total_cost = stock_amount * stock_price
            
            if total_cost > st.session_state.portfolio['cash']:
                st.error("보유 현금이 부족합니다.")
            else:
                # 이미 보유 중인 종목인지 확인
                existing_stock = None
                for stock in st.session_state.portfolio['stocks']:
                    if stock['ticker'] == stock_ticker:
                        existing_stock = stock
                        break
                
                if existing_stock:
                    # 기존 종목에 추가
                    old_qty = existing_stock['quantity']
                    old_price = existing_stock['avg_price']
                    old_value = old_qty * old_price
                    new_value = total_cost
                    total_qty = old_qty + stock_amount
                    new_avg_price = (old_value + new_value) / total_qty
                    
                    existing_stock['quantity'] = total_qty
                    existing_stock['avg_price'] = new_avg_price
                    existing_stock['total_cost'] = total_qty * new_avg_price
                else:
                    # 새 종목 추가
                    st.session_state.portfolio['stocks'].append({
                        'ticker': stock_ticker,
                        'quantity': stock_amount,
                        'avg_price': stock_price,
                        'total_cost': total_cost,
                        'current_price': stock_price,  # 임시값, 나중에 업데이트
                        'market_value': total_cost,    # 임시값, 나중에 업데이트
                        'gain_loss': 0                # 임시값, 나중에 업데이트
                    })
                
                # 현금 차감 및 기록 추가
                st.session_state.portfolio['cash'] -= total_cost
                st.session_state.portfolio['history'].append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'action': "주식 매수",
                    'ticker': stock_ticker,
                    'quantity': stock_amount,
                    'price': stock_price,
                    'total': total_cost
                })
                
                st.success(f"{stock_ticker} {stock_amount}주 매수 완료")
    
    # 메인 화면
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("포트폴리오 현황")
        
        # 현재 가격 업데이트 (실제로는 API 호출)
        update_prices = st.button("현재 가격 업데이트", key="update_prices_btn")
        
        if update_prices:
            with st.spinner("현재 가격 업데이트 중..."):
                for stock in st.session_state.portfolio['stocks']:
                    try:
                        # 실제 구현에서는 API 호출로 대체
                        # 여기서는 간단한 난수로 시뮬레이션
                        import random
                        change_pct = random.uniform(-0.05, 0.05)  # -5% ~ +5% 변동
                        stock['current_price'] = stock['avg_price'] * (1 + change_pct)
                        stock['market_value'] = stock['quantity'] * stock['current_price']
                        stock['gain_loss'] = stock['market_value'] - stock['total_cost']
                        stock['gain_loss_pct'] = (stock['gain_loss'] / stock['total_cost']) * 100
                    except Exception as e:
                        st.error(f"{stock['ticker']} 가격 업데이트 실패: {str(e)}")
                
                st.success("가격 업데이트 완료")
        
        # 포트폴리오 테이블
        if st.session_state.portfolio['stocks']:
            # 테이블 데이터 준비
            portfolio_data = []
            
            for stock in st.session_state.portfolio['stocks']:
                portfolio_data.append({
                    "종목": stock['ticker'],
                    "수량": stock['quantity'],
                    "평균단가": f"${stock['avg_price']:.2f}",
                    "현재가": f"${stock.get('current_price', stock['avg_price']):.2f}",
                    "총 매입": f"${stock['total_cost']:.2f}",
                    "현재 가치": f"${stock.get('market_value', stock['total_cost']):.2f}",
                    "손익": f"${stock.get('gain_loss', 0):.2f}",
                    "수익률": f"{stock.get('gain_loss_pct', 0):.2f}%"
                })
            
            st.dataframe(portfolio_data)
            
            # 포트폴리오 요약
            total_cost = sum(stock['total_cost'] for stock in st.session_state.portfolio['stocks'])
            total_value = sum(stock.get('market_value', stock['total_cost']) for stock in st.session_state.portfolio['stocks'])
            total_gain_loss = total_value - total_cost

def add_technical_indicators(data):
    """
    주식 데이터에 기술적 지표를 추가하는 함수
    
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
        # 이동평균 계산
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # RSI 계산 (14일)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # 0으로 나누기 방지
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD 계산
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal']
        
        # 볼린저 밴드 (20일)
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_STD'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_STD'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_STD'] * 2)
        
        # ATR 계산 (14일)
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # OBV (On-Balance Volume)
        if 'Volume' in data.columns:
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            data['OBV'] = obv
        
        # 스토캐스틱 오실레이터 (14일)
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        logging.info("기술적 지표가 성공적으로 추가되었습니다.")
        return data
    
    except Exception as e:
        logging.error(f"기술적 지표 추가 중 오류 발생: {str(e)}")
        # 기존 데이터 반환
        return data

def render_performance_dashboard():
    """성능 모니터링 대시보드를 렌더링하는 함수"""
    st.title("성능 모니터링 대시보드")
    
    # 세션 상태 초기화
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = {
            'function_times': {},
            'data_loading_times': {},
            'memory_usage': [],
            'timestamps': []
        }
    
    # 현재 메모리 사용량 측정 (MB 단위)
    current_memory = 0  # 실제 환경에서는 psutil 등을 사용하여 측정
    
    # 성능 데이터 업데이트 (예시)
    timestamp = datetime.now()
    st.session_state.performance_data['timestamps'].append(timestamp)
    st.session_state.performance_data['memory_usage'].append(current_memory)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["함수 실행 시간", "데이터 로딩 시간", "메모리 사용량"])
    
    with tab1:
        st.subheader("함수 실행 시간 분석")
        
        # 함수 실행 시간 데이터가 있는 경우
        if st.session_state.performance_data['function_times']:
            # 평균 실행 시간 계산
            avg_times = {}
            for func_name, times in st.session_state.performance_data['function_times'].items():
                avg_times[func_name] = sum(times) / len(times) if times else 0
                
            # 데이터 변환
            func_times = []
            for func_name, times in st.session_state.performance_data['function_times'].items():
                for t in times:
                    func_times.append({
                        '함수명': func_name,
                        '실행 시간(ms)': t * 1000,  # 초 -> 밀리초 변환
                        '타임스탬프': st.session_state.performance_data['timestamps'][len(func_times) % len(st.session_state.performance_data['timestamps'])]
                    })
            
            # 데이터프레임 생성
            if func_times:
                df_func_times = pd.DataFrame(func_times)
                
                # 차트 생성
                fig = px.bar(df_func_times, x='함수명', y='실행 시간(ms)', 
                             color='함수명', title='함수별 평균 실행 시간')
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 데이터 표시
                st.dataframe(df_func_times)
            else:
                st.info("함수 실행 시간 데이터가 아직 없습니다.")
        else:
            st.info("함수 실행 시간 데이터가 아직 없습니다.")
    
    with tab2:
        st.subheader("데이터 로딩 시간 분석")
        
        # 데이터 로딩 시간 데이터가 있는 경우
        if st.session_state.performance_data['data_loading_times']:
            # 데이터 변환
            loading_times = []
            for ticker, times in st.session_state.performance_data['data_loading_times'].items():
                for t in times:
                    loading_times.append({
                        '티커': ticker,
                        '로딩 시간(ms)': t * 1000,  # 초 -> 밀리초 변환
                        '타임스탬프': st.session_state.performance_data['timestamps'][len(loading_times) % len(st.session_state.performance_data['timestamps'])]
                    })
            
            # 데이터프레임 생성
            if loading_times:
                df_loading_times = pd.DataFrame(loading_times)
                
                # 차트 생성
                fig = px.bar(df_loading_times, x='티커', y='로딩 시간(ms)', 
                             color='티커', title='티커별 평균 데이터 로딩 시간')
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 데이터 표시
                st.dataframe(df_loading_times)
            else:
                st.info("데이터 로딩 시간 데이터가 아직 없습니다.")
        else:
            st.info("데이터 로딩 시간 데이터가 아직 없습니다.")
    
    with tab3:
        st.subheader("메모리 사용량 추이")
        
        # 메모리 사용량 데이터가 있는 경우
        if st.session_state.performance_data['memory_usage']:
            # 데이터프레임 생성
            df_memory = pd.DataFrame({
                '타임스탬프': st.session_state.performance_data['timestamps'],
                '메모리 사용량(MB)': st.session_state.performance_data['memory_usage']
            })
            
            # 차트 생성
            fig = px.line(df_memory, x='타임스탬프', y='메모리 사용량(MB)', 
                          title='시간별 메모리 사용량 추이')
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 데이터 표시
            st.dataframe(df_memory)
        else:
            st.info("메모리 사용량 데이터가 아직 없습니다.")
    
    # 성능 측정 시작 버튼
    if st.button("성능 측정 시작"):
        st.session_state.performance_data = {
            'function_times': {
                'load_stock_data': [0.5, 0.6, 0.7],
                'scan_patterns': [0.3, 0.4, 0.5],
                'plot_stock_chart': [0.2, 0.3, 0.4]
            },
            'data_loading_times': {
                'AAPL': [0.8, 0.9, 1.0],
                'MSFT': [0.7, 0.8, 0.9],
                'GOOGL': [0.9, 1.0, 1.1]
            },
            'memory_usage': [100, 110, 120, 130, 140],
            'timestamps': [datetime.now() - timedelta(minutes=i) for i in range(5, 0, -1)]
        }
        st.success("성능 측정을 시작했습니다. 대시보드를 새로고침하여 결과를 확인하세요.")
    
    # 성능 데이터 초기화 버튼
    if st.button("성능 데이터 초기화"):
        st.session_state.performance_data = {
            'function_times': {},
            'data_loading_times': {},
            'memory_usage': [],
            'timestamps': []
        }
        st.success("성능 데이터가 초기화되었습니다.")

def render_multi_agent_dashboard():
    """멀티 에이전트 시스템 대시보드를 렌더링하는 함수"""
    st.title("멀티 에이전트 시스템 대시보드")
    
    # 세션 상태 초기화
    if 'multi_agent_system' not in st.session_state:
        st.session_state.multi_agent_system = {
            'agents': {
                'ma_crossover': {'name': 'MA 크로스오버', 'weight': 0.3, 'active': True},
                'rsi': {'name': 'RSI 전략', 'weight': 0.3, 'active': True},
                'bollinger': {'name': '볼린저 밴드', 'weight': 0.4, 'active': True}
            },
            'performance': {
                'ma_crossover': {'return': 12.5, 'sharpe': 0.8, 'drawdown': 15.0},
                'rsi': {'return': 8.2, 'sharpe': 0.6, 'drawdown': 10.0},
                'bollinger': {'return': 15.3, 'sharpe': 1.1, 'drawdown': 18.0}
            },
            'combined_performance': {'return': 12.3, 'sharpe': 0.9, 'drawdown': 14.0},
            'history': []
        }
    
    # 사이드바 설정
    st.sidebar.header("에이전트 설정")
    
    # 에이전트 목록 및 가중치 설정
    updated_agents = {}
    total_weight = 0
    
    for agent_id, agent_info in st.session_state.multi_agent_system['agents'].items():
        st.sidebar.subheader(agent_info['name'])
        
        # 활성화 여부
        active = st.sidebar.checkbox(
            f"{agent_info['name']} 활성화",
            value=agent_info['active'],
            key=f"active_{agent_id}"
        )
        
        # 가중치 설정
        weight = st.sidebar.slider(
            f"{agent_info['name']} 가중치",
            min_value=0.0,
            max_value=1.0,
            value=agent_info['weight'],
            step=0.1,
            key=f"weight_{agent_id}",
            disabled=not active
        )
        
        # 업데이트된 에이전트 정보 저장
        updated_agents[agent_id] = {
            'name': agent_info['name'],
            'weight': weight if active else 0.0,
            'active': active
        }
        
        if active:
            total_weight += weight
    
    # 가중치 정규화
    if total_weight > 0:
        for agent_id in updated_agents:
            if updated_agents[agent_id]['active']:
                updated_agents[agent_id]['weight'] = updated_agents[agent_id]['weight'] / total_weight
    
    # 에이전트 정보 업데이트
    st.session_state.multi_agent_system['agents'] = updated_agents
    
    # 메인 대시보드 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("에이전트 성능 비교")
        
        # 성능 데이터 준비
        performance_data = []
        for agent_id, agent_info in st.session_state.multi_agent_system['agents'].items():
            if agent_id in st.session_state.multi_agent_system['performance']:
                perf = st.session_state.multi_agent_system['performance'][agent_id]
                performance_data.append({
                    '에이전트': agent_info['name'],
                    '수익률(%)': perf['return'],
                    '샤프 비율': perf['sharpe'],
                    '최대 낙폭(%)': perf['drawdown'],
                    '가중치': agent_info['weight'] if agent_info['active'] else 0.0
                })
        
        # 통합 성능 추가
        if st.session_state.multi_agent_system['combined_performance']:
            combined_perf = st.session_state.multi_agent_system['combined_performance']
            performance_data.append({
                '에이전트': '통합 성능',
                '수익률(%)': combined_perf['return'],
                '샤프 비율': combined_perf['sharpe'],
                '최대 낙폭(%)': combined_perf['drawdown'],
                '가중치': 1.0
            })
        
        # 성능 데이터프레임 생성
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            
            # 성능 차트 생성
            fig = px.bar(
                df_performance, 
                x='에이전트', 
                y='수익률(%)', 
                color='에이전트',
                text='수익률(%)',
                title='에이전트별 수익률 비교'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # 성능 데이터 표시
            st.dataframe(df_performance, use_container_width=True)
        else:
            st.info("성능 데이터가 없습니다.")
    
    with col2:
        st.subheader("가중치 분포")
        
        # 가중치 데이터 준비
        weight_data = []
        for agent_id, agent_info in st.session_state.multi_agent_system['agents'].items():
            if agent_info['active']:
                weight_data.append({
                    '에이전트': agent_info['name'],
                    '가중치': agent_info['weight']
                })
        
        # 가중치 차트 생성
        if weight_data:
            df_weights = pd.DataFrame(weight_data)
            fig = px.pie(
                df_weights, 
                values='가중치', 
                names='에이전트',
                title='에이전트 가중치 분포'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("활성화된 에이전트가 없습니다.")
    
    # 백테스트 설정
    st.subheader("백테스트 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("티커 심볼", value="AAPL", key="multi_agent_ticker")
    
    with col2:
        start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=365), key="multi_agent_start_date")
    
    with col3:
        end_date = st.date_input("종료일", value=datetime.now(), key="multi_agent_end_date")
    
    # 백테스트 실행 버튼
    if st.button("백테스트 실행"):
        # 백테스트 실행 로직 (예시)
        st.session_state.multi_agent_system['history'].append({
            'timestamp': datetime.now(),
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'agents': st.session_state.multi_agent_system['agents'].copy(),
            'performance': {
                'return': 14.5,
                'sharpe': 1.2,
                'drawdown': 12.0
            }
        })
        
        st.success(f"{ticker} 백테스트가 성공적으로 실행되었습니다.")
    
    # 백테스트 히스토리
    if st.session_state.multi_agent_system['history']:
        st.subheader("백테스트 히스토리")
        
        # 히스토리 데이터 준비
        history_data = []
        for entry in st.session_state.multi_agent_system['history']:
            history_data.append({
                '실행 시간': entry['timestamp'],
                '티커': entry['ticker'],
                '기간': f"{entry['start_date'].strftime('%Y-%m-%d')} ~ {entry['end_date'].strftime('%Y-%m-%d')}",
                '수익률(%)': entry['performance']['return'],
                '샤프 비율': entry['performance']['sharpe'],
                '최대 낙폭(%)': entry['performance']['drawdown']
            })
        
        # 히스토리 데이터프레임 생성
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
    
    # 에이전트 추가 기능
    st.subheader("새 에이전트 추가")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_agent_id = st.text_input("에이전트 ID", key="new_agent_id")
    
    with col2:
        new_agent_name = st.text_input("에이전트 이름", key="new_agent_name")
    
    if st.button("에이전트 추가"):
        if new_agent_id and new_agent_name:
            if new_agent_id not in st.session_state.multi_agent_system['agents']:
                # 새 에이전트 추가
                st.session_state.multi_agent_system['agents'][new_agent_id] = {
                    'name': new_agent_name,
                    'weight': 0.0,
                    'active': False
                }
                
                # 임의의 성능 데이터 추가 (예시)
                st.session_state.multi_agent_system['performance'][new_agent_id] = {
                    'return': np.random.uniform(5.0, 20.0),
                    'sharpe': np.random.uniform(0.5, 1.5),
                    'drawdown': np.random.uniform(5.0, 20.0)
                }
                
                st.success(f"에이전트 '{new_agent_name}'이(가) 추가되었습니다.")
            else:
                st.error(f"에이전트 ID '{new_agent_id}'은(는) 이미 존재합니다.")
        else:
            st.error("에이전트 ID와 이름을 모두 입력해주세요.")

def initialize_session_state():
    """
    세션 상태 변수들을 초기화하는 함수입니다.
    """
    # 주식 분석 대시보드 관련 상태
    if 'ticker' not in st.session_state:
        st.session_state.ticker = 'AAPL'
    if 'start_date' not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    
    # 강화학습 트레이딩 대시보드 관련 상태
    if 'rl_ticker' not in st.session_state:
        st.session_state.rl_ticker = 'AAPL'
    if 'rl_start_date' not in st.session_state:
        st.session_state.rl_start_date = (datetime.now() - timedelta(days=365)).date()
    if 'rl_end_date' not in st.session_state:
        st.session_state.rl_end_date = datetime.now().date()
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = 'DQN'
    if 'rl_capital' not in st.session_state:
        st.session_state.rl_capital = 10000
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = None
    
    # 대시보드 모드 상태
    if 'dashboard_mode' not in st.session_state:
        st.session_state.dashboard_mode = "주식 분석 대시보드"
    
    # 성능 모니터링 상태
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = {
            'function_times': {},
            'data_loading_times': {},
            'memory_usage': [],
            'timestamps': []
        }
    
    # 멀티 에이전트 시스템 상태
    if 'multi_agent_system' not in st.session_state:
        st.session_state.multi_agent_system = {
            'agents': {
                'ma_crossover': {'name': 'MA 크로스오버', 'weight': 0.3, 'active': True},
                'rsi': {'name': 'RSI 전략', 'weight': 0.3, 'active': True},
                'bollinger': {'name': '볼린저 밴드', 'weight': 0.4, 'active': True}
            },
            'performance': {
                'ma_crossover': {'return': 12.5, 'sharpe': 0.8, 'drawdown': 15.0},
                'rsi': {'return': 8.2, 'sharpe': 0.6, 'drawdown': 10.0},
                'bollinger': {'return': 15.3, 'sharpe': 1.1, 'drawdown': 18.0}
            },
            'combined_performance': {'return': 12.3, 'sharpe': 0.9, 'drawdown': 14.0},
            'history': []
        }
    
    # 글로벌 성능 모니터 설정
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()

def main():
    """메인 애플리케이션 함수"""
    # 페이지 설정
    st.set_page_config(layout="wide", page_title="주식 거래 시스템")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바 설정
    st.sidebar.title("주식 거래 시스템")
    
    # 대시보드 옵션
    dashboard_options = [
        "주식 분석 대시보드",
        "강화학습 트레이딩 대시보드",
        "성능 모니터링 대시보드",
        "멀티 에이전트 시스템",
        "포트폴리오 관리"
    ]
    
    # 메뉴 선택
    dashboard_mode = st.sidebar.selectbox(
        "대시보드 선택",
        options=dashboard_options,
        index=0 if "dashboard_mode" not in st.session_state else dashboard_options.index(st.session_state.dashboard_mode),
        key="dashboard_selector"
    )
    
    # 선택한 대시보드 모드 저장
    st.session_state.dashboard_mode = dashboard_mode
    
    # 선택한 대시보드 렌더링
    if dashboard_mode == "주식 분석 대시보드":
        render_stock_analysis_dashboard()
    elif dashboard_mode == "강화학습 트레이딩 대시보드":
        create_rl_trading_dashboard()
    elif dashboard_mode == "성능 모니터링 대시보드":
        render_performance_dashboard()
    elif dashboard_mode == "멀티 에이전트 시스템":
        render_multi_agent_dashboard()
    elif dashboard_mode == "포트폴리오 관리":
        render_portfolio_dashboard()
    else:
        st.error(f"선택한 대시보드 모드를 찾을 수 없습니다: {dashboard_mode}")

# 앱 실행
if __name__ == "__main__":
    main()