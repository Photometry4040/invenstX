"""
InvenstX - 주식 투자 분석 도구
메인 Streamlit 앱 파일
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import logging
import plotly.graph_objects as go
import traceback

# 패키지 모듈 임포트
from charts import (
    create_enhanced_chart,
    create_express_chart,
    plot_technical_indicators,
    plot_volume_chart,
    plot_pattern_chart
)

from charts.technical_charts import create_simple_indicator_chart  # 새로운 간단한 지표 차트 함수 임포트

from data import (
    load_stock_data,
    load_multiple_stocks,
    get_latest_market_data
)

from indicators import (
    calculate_technical_indicators,
    add_technical_indicators,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd
)

from indicators.patterns import scan_patterns

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 페이지 설정
st.set_page_config(
    page_title="InvenstX - 주식 투자 분석 도구",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'start_date' not in st.session_state:
    st.session_state.start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.datetime.now().strftime('%Y-%m-%d')

def render_stock_analysis_dashboard():
    """주식 분석 대시보드 렌더링"""
    st.title("주식 분석 대시보드")
    
    # 사이드바 설정
    with st.sidebar:
        st.subheader("설정")
        ticker = st.text_input("주식 티커 심볼", value=st.session_state.ticker)
        start_date = st.date_input(
            "시작 날짜",
            value=datetime.datetime.strptime(st.session_state.start_date, '%Y-%m-%d'),
            format="YYYY-MM-DD"
        )
        end_date = st.date_input(
            "종료 날짜",
            value=datetime.datetime.strptime(st.session_state.end_date, '%Y-%m-%d'),
            format="YYYY-MM-DD"
        )
        
        # 설정 업데이트 버튼
        if st.button("분석 실행"):
            st.session_state.ticker = ticker
            st.session_state.start_date = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')
            st.rerun()
    
    # 메인 컨텐츠 영역
    ticker = st.session_state.ticker
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    
    # 데이터 로드
    with st.spinner("주식 데이터 로딩 중..."):
        try:
            data = load_stock_data(ticker, start_date, end_date)
            
            if data is not None and not data.empty:
                logging.info(f"{ticker} 데이터 로드 완료: {len(data)}개 행")
                st.success(f"{ticker} 데이터 로드 완료 ({len(data)}개 행)")
                
                # 데이터프레임 구조 확인 및 재구성 (필요한 경우)
                try:
                    # 데이터프레임 구조 검사
                    logging.info(f"데이터프레임 구조: {type(data)}, 컬럼: {data.columns.tolist()}, 인덱스 타입: {type(data.index)}")
                    
                    # 다중 인덱스 확인
                    if isinstance(data.index, pd.MultiIndex):
                        logging.warning("다중 인덱스 감지됨. 단일 인덱스로 재구성합니다.")
                        # 데이터프레임 재구성
                        data = data.reset_index()
                    
                    # 다중 레벨 컬럼 확인 및 처리
                    if isinstance(data.columns, pd.MultiIndex):
                        logging.warning("다중 레벨 컬럼 감지됨. 단일 레벨로 변환합니다.")
                        # 다중 레벨 컬럼을 단일 레벨로 변환
                        new_columns = []
                        for col in data.columns:
                            # 튜플 컬럼을 문자열로 변환 (예: ('Close', 'AAPL') -> 'Close')
                            if isinstance(col, tuple):
                                new_columns.append(col[0])  # 첫 번째 레벨만 사용
                            else:
                                new_columns.append(col)
                                
                        # 새 컬럼명으로 데이터프레임 생성
                        data.columns = new_columns
                        logging.info(f"컬럼 변환 완료: {data.columns.tolist()}")
                    
                    # 컬럼 구조 확인
                    try:
                        for col in data.columns:
                            # 각 열에 대해 단일 값 접근 시도
                            val_check = data.iloc[0][col]
                            logging.info(f"컬럼 {col}의 첫 번째 값 타입: {type(val_check)}")
                    except Exception as col_err:
                        logging.error(f"컬럼 구조 확인 중 오류: {str(col_err)}")
                except Exception as struct_err:
                    logging.error(f"데이터프레임 구조 검사 중 오류: {str(struct_err)}")
                
                # 데이터 유효성 검사 추가
                required_columns = ['Open', 'High', 'Low', 'Close']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"필수 데이터 열이 누락되었습니다: {', '.join(missing_columns)}")
                    logging.error(f"차트 생성 취소: 필수 열 누락 - {missing_columns}")
                    return
                
                # 이 부분에서 차트를 생성하기 전에 데이터 컬럼을 안전하게 처리
                try:
                    # 새로운 데이터프레임 생성
                    safe_data = pd.DataFrame()
                    
                    for col in data.columns:
                        try:
                            # 각 컬럼을 안전하게 추출하여 새 데이터프레임에 추가
                            col_values = data[col].values  # NumPy 배열로 변환
                            safe_data[col] = col_values
                            
                            # 숫자형으로 변환 시도
                            if col in required_columns:
                                safe_data[col] = pd.to_numeric(safe_data[col], errors='coerce')
                                logging.info(f"{col} 열을 숫자형으로 변환했습니다.")
                        except Exception as e:
                            logging.error(f"{col} 열 처리 중 오류: {str(e)}")
                    
                    # 인덱스 복사
                    safe_data.index = data.index
                    
                    # 원본 데이터프레임 대체
                    data = safe_data
                    logging.info("데이터프레임 안전하게 재구성 완료")
                except Exception as df_err:
                    logging.error(f"데이터프레임 재구성 중 오류: {str(df_err)}")
                
                # 주식 차트 표시
                st.subheader("주가 차트")
                try:
                    try:
                        # create_express_chart 함수 사용 시도
                        fig = create_express_chart(data, ticker)
                        logging.info("Plotly Express로 차트 생성 시도")
                    except Exception as express_err:
                        logging.error(f"Express 차트 생성 오류: {str(express_err)}")
                        # 실패 시 create_enhanced_chart 함수 사용 시도
                        fig = create_enhanced_chart(data, ticker)
                        logging.info("Enhanced 차트로 대체 생성 시도")
                    
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                        logging.info("차트 표시 성공")
                    else:
                        st.error("차트 생성에 실패했습니다.")
                        logging.error("차트 객체가 None입니다.")
                except Exception as chart_err:
                    st.error(f"차트 생성 중 오류 발생: {str(chart_err)}")
                    logging.error(f"차트 생성 오류: {str(chart_err)}")
                    # Fallback - 기본 라인 차트 시도
                    try:
                        st.warning("기본 차트로 대체하여 표시합니다.")
                        st.line_chart(data['Close'])
                        logging.info("기본 라인 차트로 대체 표시")
                    except Exception as fallback_err:
                        st.error("모든 차트 방식에 실패했습니다.")
                        logging.error(f"모든 차트 방식 실패: {str(fallback_err)}")
                
                # 기술적 지표 차트 표시
                st.subheader("📈 기술적 지표")
                try:
                    # 기술적 지표 추가
                    data_with_indicators = add_technical_indicators(data)
                    # 새로운 간단한 기술적 지표 차트 함수 사용
                    tech_fig = create_simple_indicator_chart(data_with_indicators, ticker)
                    st.plotly_chart(tech_fig, use_container_width=True)
                    logging.info("간단한 기술적 지표 차트 생성 완료")
                except Exception as indicator_error:
                    error_msg = str(indicator_error)
                    logging.error(f"기술적 지표 차트 생성 중 오류 발생: {error_msg}")
                    st.error(f"기술적 지표 차트를 생성할 수 없습니다: {error_msg}")
                    logging.debug(traceback.format_exc())
                
                # 패턴 스캐너
                st.subheader("🔍 패턴 스캐너")
                try:
                    pattern_results = scan_patterns(data_with_indicators)
                    if pattern_results is not None and not pattern_results.empty:
                        detected_patterns = [col for col in pattern_results.columns 
                                           if pattern_results[col].sum() > 0]
                        
                        if detected_patterns:
                            st.success(f"감지된 패턴: {', '.join(detected_patterns)}")
                            
                            # 패턴 선택 드롭다운
                            selected_pattern = st.selectbox(
                                "패턴 선택", 
                                options=detected_patterns, 
                                index=0
                            )
                            
                            # 선택된 패턴으로 차트 생성
                            try:
                                pattern_chart = plot_pattern_chart(data_with_indicators, pattern_results, selected_pattern)
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
        except Exception as load_e:
            st.error(f"데이터 로드 중 오류 발생: {str(load_e)}")
            logging.error(f"데이터 로드 오류: {str(load_e)}")
            logging.debug(traceback.format_exc())

def main():
    # 네비게이션
    st.sidebar.title("InvenstX")
    page = st.sidebar.radio("페이지 선택", ["주식 분석 대시보드", "설정", "도움말"])
    
    if page == "주식 분석 대시보드":
        render_stock_analysis_dashboard()
    elif page == "설정":
        st.title("설정")
        st.write("앱 설정이 여기에 표시됩니다.")
    elif page == "도움말":
        st.title("도움말")
        st.write("InvenstX 사용 방법 및 도움말이 여기에 표시됩니다.")

if __name__ == "__main__":
    main() 