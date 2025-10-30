"""
기술적 지표 차트 생성 관련 함수
MACD, RSI, 볼린저 밴드 등의 기술적 지표 차트 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import traceback

def plot_technical_indicators(data, indicators=None, ticker=''):
    """
    기술적 지표 차트 생성
    
    Args:
        data (pd.DataFrame): 주식 데이터
        indicators (list, optional): 표시할 기술적 지표 목록
        ticker (str, optional): 종목 코드
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("기술적 지표 차트 생성 실패: 데이터가 비어 있습니다.")
            return go.Figure().update_layout(title="데이터가 없습니다.")
        
        # 필수 컬럼 확인
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"기술적 지표 차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            return go.Figure().update_layout(title=f"필수 데이터가 누락되었습니다: {', '.join(missing_columns)}")
        
        # 데이터가 충분한지 확인
        if len(data) < 50:  # 대부분의 기술적 지표는 충분한 데이터 필요
            logging.warning(f"기술적 지표 차트 생성 주의: 데이터가 충분하지 않을 수 있습니다 ({len(data)}행)")
        
        # 데이터 타입 확인 및 변환
        data_copy = data.copy()
        
        # 모든 필요한 열에 대해 데이터 타입 확인 및 변환
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA60', 'MA120', 'RSI', 'MACD', 'Signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
        
        for col in numeric_columns:
            if col in data_copy.columns:
                try:
                    if not pd.api.types.is_numeric_dtype(data_copy[col]):
                        logging.warning(f"기술적 지표 차트: {col} 열이 숫자형이 아닙니다. 현재 타입: {data_copy[col].dtype}")
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                        logging.info(f"{col} 열을 숫자형으로 변환했습니다.")
                except Exception as e:
                    logging.error(f"{col} 열을 숫자형으로 변환하는 중 오류 발생: {str(e)}")
                    logging.debug(f"{col} 열의 처음 5개 값: {data_copy[col].head().tolist() if col in data_copy.columns else 'column not found'}")

        # NaN 값이 있는지 확인
        for col in [c for c in numeric_columns if c in data_copy.columns]:
            if data_copy[col].isna().any():
                nan_count = data_copy[col].isna().sum()
                logging.warning(f"{col} 열에 {nan_count}개의 NaN 값이 있습니다.")
        
        # 기본 지표 설정
        if indicators is None:
            indicators = ['MACD', 'RSI', 'Bollinger Bands']
        
        # 기술적 지표가 있는지 확인
        available_indicators = []
        
        # MACD 지표 확인
        macd_cols = ['MACD', 'MACD_Signal', 'MACD_Hist']
        if 'MACD' in indicators and all(col in data_copy.columns for col in macd_cols):
            available_indicators.append('MACD')
        
        # RSI 지표 확인
        if 'RSI' in indicators and 'RSI' in data_copy.columns:
            available_indicators.append('RSI')
        
        # 볼린저 밴드 확인
        bb_cols = ['BB_Upper', 'BB_Middle', 'BB_Lower']
        if 'Bollinger Bands' in indicators and all(col in data_copy.columns for col in bb_cols):
            available_indicators.append('Bollinger Bands')
        
        # 이용 가능한 지표가 없는 경우
        if not available_indicators:
            logging.warning("기술적 지표 차트 생성 실패: 표시할 기술적 지표가 없습니다.")
            return go.Figure().update_layout(title="표시할 기술적 지표가 없습니다")
        
        # 데이터 복사본 생성하여 원본 데이터 보존
        data_copy = data.copy()
        
        # 서브플롯 개수 설정
        subplot_count = len(available_indicators)
        
        # MACD+RSI: 2개 차트, Bollinger Bands: 가격 차트와 함께 표시
        if 'Bollinger Bands' in available_indicators:
            subplot_count -= 1
            bb_chart = True
        else:
            bb_chart = False
        
        # 가격 데이터와 BB를 표시할 첫 번째 차트 추가
        subplot_count += 1
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=subplot_count, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['가격 & 볼린저 밴드'] + [ind for ind in available_indicators if ind != 'Bollinger Bands']
        )
        
        # 가격 차트 추가 (첫 번째 서브플롯)
        if 'Close' in data_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['Close'],
                    line=dict(color='black', width=1),
                    name="종가",
                    legendgroup='price'
                ),
                row=1, col=1
            )
        
        # 볼린저 밴드 추가 (가격 차트와 함께)
        if 'Bollinger Bands' in available_indicators:
            # 상단 밴드
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['BB_Upper'],
                    line=dict(color='red', width=1, dash='dash'),
                    name="BB 상단",
                    legendgroup='bollinger'
                ),
                row=1, col=1
            )
            
            # 중간 밴드 (20일 이동평균)
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['BB_Middle'],
                    line=dict(color='blue', width=1),
                    name="BB 중간 (20일 MA)",
                    legendgroup='bollinger'
                ),
                row=1, col=1
            )
            
            # 하단 밴드
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['BB_Lower'],
                    line=dict(color='green', width=1, dash='dash'),
                    name="BB 하단",
                    legendgroup='bollinger'
                ),
                row=1, col=1
            )
        
        # 현재 서브플롯 인덱스
        current_subplot = 2
        
        # MACD 추가
        if 'MACD' in available_indicators:
            # MACD 라인
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['MACD'],
                    line=dict(color='blue', width=1),
                    name="MACD",
                    legendgroup='macd'
                ),
                row=current_subplot, col=1
            )
            
            # MACD 시그널 라인
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['MACD_Signal'],
                    line=dict(color='red', width=1),
                    name="MACD 시그널",
                    legendgroup='macd'
                ),
                row=current_subplot, col=1
            )
            
            # MACD 히스토그램
            colors = []
            for val in data_copy['MACD_Hist']:
                if val >= 0:
                    colors.append('rgba(0, 255, 0, 0.5)')  # 초록색 (양수)
                else:
                    colors.append('rgba(255, 0, 0, 0.5)')  # 빨간색 (음수)
            
            fig.add_trace(
                go.Bar(
                    x=data_copy.index,
                    y=data_copy['MACD_Hist'],
                    marker_color=colors,
                    name="MACD 히스토그램",
                    legendgroup='macd'
                ),
                row=current_subplot, col=1
            )
            
            # 0선 추가
            fig.add_shape(
                type="line",
                x0=data_copy.index[0],
                y0=0,
                x1=data_copy.index[-1],
                y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=current_subplot, col=1
            )
            
            current_subplot += 1
        
        # RSI 추가
        if 'RSI' in available_indicators:
            # RSI 라인
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index,
                    y=data_copy['RSI'],
                    line=dict(color='purple', width=1),
                    name="RSI",
                    legendgroup='rsi'
                ),
                row=current_subplot, col=1
            )
            
            # 과매수/과매도 영역 표시
            # 과매수 선 (70)
            fig.add_shape(
                type="line",
                x0=data_copy.index[0],
                y0=70,
                x1=data_copy.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=current_subplot, col=1
            )
            
            # 과매도 선 (30)
            fig.add_shape(
                type="line",
                x0=data_copy.index[0],
                y0=30,
                x1=data_copy.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=current_subplot, col=1
            )
            
            # 중간선 (50)
            fig.add_shape(
                type="line",
                x0=data_copy.index[0],
                y0=50,
                x1=data_copy.index[-1],
                y1=50,
                line=dict(color="black", width=1, dash="dash"),
                row=current_subplot, col=1
            )
            
            current_subplot += 1
        
        # 차트 레이아웃 설정
        ticker_title = f"{ticker} " if ticker else ""
        fig.update_layout(
            title=f"{ticker_title}기술적 지표 차트",
            xaxis_title="날짜",
            height=200 * subplot_count,  # 서브플롯 개수에 따라 높이 조정
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
        
        # Y축 레이블 설정
        if subplot_count >= 1:
            fig.update_yaxes(title_text="가격", row=1, col=1)
        
        if 'MACD' in available_indicators:
            macd_row = 2 if not bb_chart else 1
            fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
        
        if 'RSI' in available_indicators:
            rsi_row = 3 if 'MACD' in available_indicators and not bb_chart else 2
            if bb_chart and 'MACD' not in available_indicators:
                rsi_row = 1
            fig.update_yaxes(title_text="RSI", row=rsi_row, col=1)
        
        return fig
    except Exception as e:
        logging.error(f"기술적 지표 차트 생성 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환하여 앱이 중단되지 않도록 함
        return go.Figure().update_layout(title=f"기술적 지표 차트 생성 중 오류: {str(e)}")

def create_simple_indicator_chart(data, ticker=''):
    """
    단순화된 기술적 지표 차트 생성
    
    Args:
        data (pd.DataFrame): 주식 데이터 (기술적 지표 포함)
        ticker (str, optional): 종목 코드
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            logging.error("기술적 지표 차트 생성 실패: 데이터가 비어 있습니다.")
            fig = px.line()
            fig.add_annotation(text="데이터가 없습니다", showarrow=False, font=dict(size=20))
            return fig
        
        # 필수 컬럼 확인
        if 'Close' not in data.columns:
            logging.error("기술적 지표 차트 생성 실패: Close 컬럼이 필요합니다")
            fig = px.line()
            fig.add_annotation(text="Close 컬럼이 없습니다", showarrow=False, font=dict(size=20))
            return fig
        
        # 데이터 복사본 생성
        df = data.copy()
        
        # 데이터 타입 확인 및 변환
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and col != 'Date':
                try:
                    logging.warning(f"{col} 열이 숫자형이 아닙니다. 변환을 시도합니다.")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logging.error(f"{col} 변환 오류: {str(e)}")
        
        # 필요한 기술적 지표 계산 (없는 경우)
        if 'RSI' not in df.columns:
            # RSI 계산
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)  # 0으로 나누기 방지
            df['RSI'] = 100 - (100 / (1 + rs))
        
        if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
            # MACD 계산
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        if 'BB_Upper' not in df.columns or 'BB_Middle' not in df.columns or 'BB_Lower' not in df.columns:
            # 볼린저 밴드 계산
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        
        # 서브플롯 생성 (3개 차트: 가격+볼린저, MACD, RSI)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=["가격 & 볼린저 밴드", "MACD", "RSI"]
        )
        
        # 1. 가격 차트 + 볼린저 밴드
        # 종가 라인
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['Close'],
                mode='lines',
                name='종가',
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
        
        # 볼린저 밴드
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['BB_Upper'],
                mode='lines',
                name='BB 상단',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['BB_Middle'],
                mode='lines',
                name='BB 중간',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['BB_Lower'],
                mode='lines',
                name='BB 하단',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. MACD 차트
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MACD_Signal'],
                mode='lines',
                name='신호선',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # MACD 히스토그램
        colors = np.where(df['MACD_Hist'] >= 0, 'rgba(0,255,0,0.5)', 'rgba(255,0,0,0.5)')
        
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['MACD_Hist'],
                name='MACD 히스토그램',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # 0선
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=0,
            x1=df.index[-1],
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=2, col=1
        )
        
        # 3. RSI 차트
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # 과매수/과매도 라인
        for level, color in [(30, 'green'), (50, 'black'), (70, 'red')]:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=level,
                x1=df.index[-1],
                y1=level,
                line=dict(color=color, width=1, dash="dash"),
                row=3, col=1
            )
        
        # 차트 레이아웃 설정
        ticker_title = f"{ticker} " if ticker else ""
        fig.update_layout(
            title=f"{ticker_title}기술적 지표 차트",
            height=800,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Y축 레이블 설정
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    except Exception as e:
        logging.error(f"기술적 지표 차트 생성 중 오류 발생: {str(e)}")
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환
        fig = px.line()
        fig.add_annotation(text=f"차트 생성 중 오류: {str(e)}", showarrow=False)
        return fig 