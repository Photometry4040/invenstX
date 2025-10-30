"""
주식 차트 생성 관련 함수
캔들스틱 차트, 이동평균선, 거래량 등 표시
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

def create_enhanced_chart(data, ticker):
    """
    개선된 주식 차트 생성
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        ticker (str): 종목 코드
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            # 누락된 컬럼 중 필수 컬럼이 있는지 확인
            essential_columns = ['Open', 'High', 'Low', 'Close']
            essential_missing = [col for col in essential_columns if col in missing_columns]
            
            if essential_missing:
                # 필수 컬럼이 누락된 경우 차트 생성 불가
                return go.Figure().update_layout(title=f"필수 데이터가 누락되었습니다: {', '.join(essential_missing)}")
            
            # Volume만 누락되었다면 더미 데이터 생성
            if 'Volume' in missing_columns:
                data = data.copy()
                data['Volume'] = np.zeros(len(data))
                logging.warning("거래량 데이터가 누락되어 0으로 채웠습니다.")
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            logging.warning(f"차트 생성 실패: 데이터가 충분하지 않습니다 ({len(data)}행)")
            return go.Figure().update_layout(title="데이터가 충분하지 않습니다")
        
        # 데이터 복사본 생성하여 원본 데이터 보존
        data_copy = data.copy()
        
        # 데이터 타입 확인 및 변환
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in data_copy.columns:
                try:
                    # 수정: DataFrame.dtype이 아닌 Series.dtype 사용
                    # 열이 숫자형이 아닌 경우에만 변환 시도
                    if not pd.api.types.is_numeric_dtype(data_copy[col]):
                        logging.warning(f"{col} 열이 숫자형이 아닙니다. 현재 타입: {data_copy[col].dtype}")
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                        logging.info(f"{col} 열을 숫자형으로 변환했습니다.")
                except Exception as type_e:
                    logging.error(f"{col} 열을 숫자형으로 변환하는 중 오류 발생: {str(type_e)}")
                    # 변환 실패 시 디버그 정보 로깅
                    logging.debug(f"{col} 열의 처음 5개 값: {data_copy[col].head().tolist()}")
        
        # NaN 값 처리
        essential_columns = ['Open', 'High', 'Low', 'Close']
        essential_nan = data_copy[essential_columns].isna().any(axis=1)
        
        if essential_nan.any():
            logging.warning(f"필수 열에 NaN 값이 있습니다. NaN 데이터 수: {essential_nan.sum()}")
            # NaN이 있는 행 제거 (필수 컬럼에 한해서)
            data_copy = data_copy.dropna(subset=essential_columns)
            logging.info(f"NaN 값 제거 후 {len(data_copy)}행 남음")
            
            if len(data_copy) < 5:
                logging.error("NaN 값 제거 후 데이터가 부족합니다.")
                return go.Figure().update_layout(title="유효한 데이터가 부족합니다 (NaN 값이 너무 많음)")
        
        # Volume의 NaN 값은 0으로 대체
        if 'Volume' in data_copy.columns and data_copy['Volume'].isna().any():
            data_copy['Volume'] = data_copy['Volume'].fillna(0)
            logging.warning("거래량의 NaN 값을 0으로 대체했습니다.")
        
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
        
        # 이동평균선 추가
        ma20 = data_copy['Close'].rolling(window=20).mean()
        ma50 = data_copy['Close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=ma20,
                line=dict(color='blue', width=1),
                name="20일 이동평균"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=ma50,
                line=dict(color='red', width=1),
                name="50일 이동평균"
            )
        )
        
        # 거래량 차트 (secondary y-axis)
        fig.add_trace(
            go.Bar(
                x=data_copy.index,
                y=data_copy['Volume'],
                name="거래량",
                marker_color='rgba(0, 0, 255, 0.5)',
                yaxis="y2"  # 보조 y축 사용
            )
        )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title=f"{ticker} 주가 및 거래량 차트",
            xaxis_title="날짜",
            yaxis_title="가격",
            yaxis2=dict(
                title="거래량",
                title_font=dict(color="rgba(0, 0, 255, 0.5)"),
                tickfont=dict(color="rgba(0, 0, 255, 0.5)"),
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
        logging.error(f"차트 생성 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환하여 앱이 중단되지 않도록 함
        return go.Figure().update_layout(title=f"차트 생성 중 오류: {str(e)}")

def create_express_chart(data, ticker):
    """
    Plotly Express를 사용한 주식 차트 생성
    
    Args:
        data (pd.DataFrame): OHLCV 형식의 주식 데이터
        ticker (str): 종목 코드
        
    Returns:
        go.Figure: Plotly 차트 객체
    """
    try:
        # 데이터 유효성 검사
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"차트 생성 실패: 필요한 컬럼이 누락되었습니다 - {missing_columns}")
            # 필수 컬럼이 누락되었는지 확인
            essential_columns = ['Open', 'High', 'Low', 'Close']
            essential_missing = [col for col in essential_columns if col in missing_columns]
            
            if essential_missing:
                # 필수 컬럼이 없으면 오류 메시지와 함께 빈 차트 반환
                fig = px.line()
                fig.add_annotation(text=f"필수 데이터가 누락되었습니다: {', '.join(essential_missing)}", 
                                  showarrow=False, font=dict(size=20))
                return fig
            
            # Volume만 누락된 경우 더미 데이터 생성
            if 'Volume' in missing_columns:
                data = data.copy()
                data['Volume'] = np.zeros(len(data))
                logging.warning("거래량 데이터가 누락되어 0으로 채웠습니다.")
        
        # 데이터가 충분한지 확인
        if len(data) < 5:
            logging.warning(f"차트 생성 실패: 데이터가 충분하지 않습니다 ({len(data)}행)")
            fig = px.line()
            fig.add_annotation(text="데이터가 충분하지 않습니다", showarrow=False, font=dict(size=20))
            return fig
        
        # 데이터 복사본 생성
        df = data.copy()
        
        # 데이터가 인덱스가 아닌 열로 날짜를 가지고 있다면 처리
        if not isinstance(df.index, pd.DatetimeIndex) and 'Date' in df.columns:
            df = df.set_index('Date')
        
        # 모든 열이 숫자형인지 확인하고 변환 시도
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logging.warning(f"{col} 열이 숫자형이 아닙니다. 변환을 시도합니다.")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logging.error(f"{col} 변환 오류: {str(e)}")
        
        # NaN 값 처리
        essential_columns = ['Open', 'High', 'Low', 'Close']
        if df[essential_columns].isna().any().any():
            logging.warning("필수 열에 NaN 값이 있습니다. 보간을 시도합니다.")
            df[essential_columns] = df[essential_columns].interpolate(method='linear', axis=0)
        
        # 이동평균선 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # 차트 생성을 위해 데이터 준비 - 인덱스를 열로 변환 (Plotly Express용)
        df_reset = df.reset_index()
        date_col = 'index' if 'index' in df_reset.columns else 'Date'
        
        # 서브플롯으로 캔들스틱과 거래량 차트 생성
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.02, 
                           row_heights=[0.7, 0.3])
        
        # 캔들스틱 추가
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='캔들스틱'
        ), row=1, col=1)
        
        # 이동평균선 추가
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA20'],
            line=dict(color='blue', width=1),
            name='20일 이동평균'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA50'],
            line=dict(color='red', width=1),
            name='50일 이동평균'
        ), row=1, col=1)
        
        # 거래량 차트 추가
        colors = ['rgba(0,0,255,0.7)' if row['Close'] >= row['Open'] else 'rgba(255,0,0,0.7)' 
                 for _, row in df.iterrows()]
        
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            marker_color=colors,
            name='거래량'
        ), row=2, col=1)
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title=f"{ticker} 주가 및 거래량 차트",
            xaxis_title="날짜",
            yaxis_title="가격",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        logging.error(f"Plotly Express 차트 생성 중 오류 발생: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        # 오류가 발생해도 빈 차트를 반환하여 앱이 중단되지 않도록 함
        fig = px.line()
        fig.add_annotation(text=f"차트 생성 중 오류: {str(e)}", showarrow=False, font=dict(size=15))
        return fig 