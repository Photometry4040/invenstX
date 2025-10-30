"""
거래량 차트 생성 관련 함수
거래량 막대 차트 및 이동평균선 표시
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import traceback

def plot_volume_chart(data, ticker=''):
    """
    거래량 차트 생성
    
    Args:
        data (pd.DataFrame): 주식 데이터 (Volume 열 포함)
        ticker (str, optional): 종목 코드
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("거래량 차트 생성 실패: 데이터가 비어 있습니다.")
            return go.Figure().update_layout(title="거래량 데이터가 없습니다.")
        
        # Volume 컬럼 확인
        if 'Volume' not in data.columns:
            logging.error("거래량 차트 생성 실패: Volume 컬럼이 누락되었습니다.")
            return go.Figure().update_layout(title="Volume 데이터가 누락되었습니다")
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            logging.warning(f"거래량 차트 생성 실패: 데이터가 충분하지 않습니다 ({len(data)}행)")
            return go.Figure().update_layout(title="데이터가 충분하지 않습니다")
        
        # 데이터 복사본 생성하여 원본 데이터 보존
        data_copy = data.copy()
        
        # NaN 값이 있는지 확인
        if data_copy['Volume'].isna().any():
            logging.warning("거래량에 NaN 값이 있습니다. 0으로 대체합니다.")
            data_copy['Volume'] = data_copy['Volume'].fillna(0)
        
        # 데이터 타입 확인
        if not np.issubdtype(data_copy['Volume'].dtype, np.number):
            try:
                data_copy['Volume'] = pd.to_numeric(data_copy['Volume'], errors='coerce')
                data_copy['Volume'] = data_copy['Volume'].fillna(0)
                logging.warning("Volume 열이 숫자형으로 변환되었습니다.")
            except Exception as type_e:
                logging.error(f"Volume 열을 숫자형으로 변환하는 중 오류 발생: {str(type_e)}")
                return go.Figure().update_layout(title="Volume 데이터를 변환할 수 없습니다")
        
        # 거래량 이동평균 계산
        data_copy['Volume_MA10'] = data_copy['Volume'].rolling(window=10).mean()
        
        # 색상 설정 (상승/하락에 따라)
        colors = []
        for i in range(len(data_copy)):
            if 'Close' in data_copy.columns and 'Open' in data_copy.columns:
                if i > 0 and data_copy['Close'].iloc[i] > data_copy['Close'].iloc[i-1]:
                    colors.append('rgba(0, 255, 0, 0.5)')  # 초록색 (상승)
                else:
                    colors.append('rgba(255, 0, 0, 0.5)')  # 빨간색 (하락)
            else:
                # Close, Open 데이터가 없는 경우 기본 색상 사용
                colors.append('rgba(0, 0, 255, 0.5)')  # 파란색
        
        # 차트 생성
        fig = go.Figure()
        
        # 거래량 막대 차트
        fig.add_trace(
            go.Bar(
                x=data_copy.index,
                y=data_copy['Volume'],
                marker_color=colors,
                name="거래량"
            )
        )
        
        # 거래량 이동평균선
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy['Volume_MA10'],
                line=dict(color='purple', width=2),
                name="10일 이동평균 거래량"
            )
        )
        
        # 차트 레이아웃 설정
        ticker_title = f"{ticker} " if ticker else ""
        fig.update_layout(
            title=f"{ticker_title}거래량 차트",
            xaxis_title="날짜",
            yaxis_title="거래량",
            height=400,
            template="plotly_white",
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
        
        # Y축 설정 (천 단위 구분)
        fig.update_yaxes(
            tickformat=",",  # 천 단위 구분자 추가
        )
        
        return fig
    except Exception as e:
        logging.error(f"거래량 차트 생성 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환하여 앱이 중단되지 않도록 함
        return go.Figure().update_layout(title=f"거래량 차트 생성 중 오류: {str(e)}") 