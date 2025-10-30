"""
캔들스틱 패턴 차트 생성 관련 함수
Doji, Hammer, Engulfing 등의 패턴을 차트에 표시
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback

def plot_pattern_chart(data, pattern_results, pattern_name=None):
    """
    패턴 감지 결과를 포함한 주식 차트 생성
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        pattern_results (pd.DataFrame): 패턴 감지 결과
        pattern_name (str, optional): 표시할 특정 패턴 이름
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("패턴 차트 생성 실패: 데이터가 비어 있습니다.")
            return go.Figure().update_layout(title="데이터가 없습니다.")
        
        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"패턴 차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            return go.Figure().update_layout(title=f"필수 데이터가 누락되었습니다: {', '.join(missing_columns)}")
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            logging.warning(f"패턴 차트 생성 실패: 데이터가 충분하지 않습니다 ({len(data)}행)")
            return go.Figure().update_layout(title="데이터가 충분하지 않습니다")
        
        # 패턴 결과 유효성 검사
        if pattern_results is None or pattern_results.empty:
            logging.warning("패턴 차트 생성 실패: 패턴 감지 결과가 없습니다.")
            return go.Figure().update_layout(title="감지된 패턴이 없습니다")
        
        # 패턴 필터링
        available_patterns = ['Doji', 'Hammer', 'Engulfing Bullish', 'Engulfing Bearish']
        
        # 특정 패턴이 지정된 경우 해당 패턴만 필터링
        if pattern_name is not None:
            if pattern_name not in available_patterns:
                logging.error(f"패턴 차트 생성 실패: 알 수 없는 패턴 - {pattern_name}")
                return go.Figure().update_layout(title=f"알 수 없는 패턴: {pattern_name}")
            
            # 패턴 결과 필터링
            for pattern in available_patterns:
                if pattern != pattern_name and pattern in pattern_results.columns:
                    pattern_results[pattern] = False
        
        # 데이터 복사본 생성하여 원본 데이터 보존
        data_copy = data.copy()
        
        # 차트 생성
        fig = go.Figure()
        
        # 캔들스틱 추가
        fig.add_trace(
            go.Candlestick(
                x=data_copy.index,
                open=data_copy['Open'],
                high=data_copy['High'],
                low=data_copy['Low'],
                close=data_copy['Close'],
                name="캔들스틱"
            )
        )
        
        # 이동평균선 추가 (있는 경우)
        if 'MA20' in data_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['MA20'],
                    line=dict(color='blue', width=1),
                    name="20일 이동평균"
                )
            )
        
        if 'MA50' in data_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['MA50'],
                    line=dict(color='red', width=1),
                    name="50일 이동평균"
                )
            )
        
        # 패턴 마커 설정
        colors = {
            'Doji': 'rgba(255, 255, 0, 0.8)',        # 노란색
            'Hammer': 'rgba(0, 255, 0, 0.8)',        # 초록색
            'Engulfing Bullish': 'rgba(0, 128, 0, 0.8)',  # 진한 초록색
            'Engulfing Bearish': 'rgba(255, 0, 0, 0.8)'   # 빨간색
        }
        
        symbols = {
            'Doji': 'star',
            'Hammer': 'triangle-up',
            'Engulfing Bullish': 'circle',
            'Engulfing Bearish': 'circle'
        }
        
        # 각 패턴에 대한 마커 추가
        for pattern in available_patterns:
            if pattern in pattern_results.columns:
                pattern_dates = pattern_results.index[pattern_results[pattern]]
                if not pattern_dates.empty:
                    logging.info(f"{pattern} 패턴이 {len(pattern_dates)}개 감지되었습니다.")
                    
                    # 패턴이 발생한 날짜의 고가에 마커 표시
                    pattern_highs = []
                    pattern_x = []
                    
                    for date in pattern_dates:
                        if date in data_copy.index:
                            # date 인덱스 위치 찾기
                            idx = data_copy.index.get_loc(date)
                            
                            # 마커 위치 계산 (고가 바로 위)
                            if pattern in ['Engulfing Bearish', 'Doji']:
                                # 베어리시 패턴은 캔들 위에 표시
                                high_value = data_copy['High'].iloc[idx]
                                if isinstance(high_value, pd.Series):
                                    high_value = high_value.iloc[0]  # Series인 경우 첫 번째 값 사용
                                marker_y = float(high_value) * 1.01
                            else:
                                # 불리시 패턴은 캔들 아래에 표시
                                low_value = data_copy['Low'].iloc[idx]
                                if isinstance(low_value, pd.Series):
                                    low_value = low_value.iloc[0]  # Series인 경우 첫 번째 값 사용
                                marker_y = float(low_value) * 0.99
                            
                            pattern_highs.append(marker_y)
                            pattern_x.append(date)
                    
                    if pattern_x:
                        fig.add_trace(
                            go.Scatter(
                                x=pattern_x,
                                y=pattern_highs,
                                mode='markers',
                                marker=dict(
                                    color=colors.get(pattern, 'black'),
                                    size=12,
                                    symbol=symbols.get(pattern, 'circle'),
                                    line=dict(width=2, color='black')
                                ),
                                name=pattern
                            )
                        )
        
        # 거래량 차트 추가 (있는 경우)
        if 'Volume' in data_copy.columns:
            colors = []
            for i in range(len(data_copy)):
                if i > 0 and data_copy['Close'].iloc[i] > data_copy['Close'].iloc[i-1]:
                    colors.append('rgba(0, 255, 0, 0.5)')  # 초록색 (상승)
                else:
                    colors.append('rgba(255, 0, 0, 0.5)')  # 빨간색 (하락)
            
            fig.add_trace(
                go.Bar(
                    x=data_copy.index,
                    y=data_copy['Volume'],
                    marker_color=colors,
                    name="거래량",
                    yaxis="y2"  # 보조 y축 사용
                )
            )
        
        # 차트 레이아웃 설정
        pattern_title = f" - {pattern_name}" if pattern_name else ""
        fig.update_layout(
            title=f"패턴 분석 차트{pattern_title}",
            xaxis_title="날짜",
            yaxis_title="가격",
            yaxis2=dict(
                title=dict(
                    text="거래량",
                    font=dict(color="rgba(0, 0, 0, 0.5)")
                ),
                tickfont=dict(color="rgba(0, 0, 0, 0.5)"),
                anchor="x",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            height=600,
            xaxis_rangeslider_visible=False,
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
        
        return fig
    except Exception as e:
        logging.error(f"패턴 차트 생성 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환하여 앱이 중단되지 않도록 함
        return go.Figure().update_layout(title=f"패턴 차트 생성 중 오류: {str(e)}") 