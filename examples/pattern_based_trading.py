"""
패턴 기반 트레이딩 예시

주식 차트 패턴을 활용한 트레이딩 전략 예시입니다.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.fetch_data import fetch_stock_data
from indicators.patterns.bollinger_bands import BollingerBands
from indicators.patterns.double_bottom import DoubleBottom
from indicators.patterns.pattern_analyzer import PatternAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pattern_analysis(ticker: str, start_date: str, end_date: str, investment_style: str = 'long_term') -> None:
    """
    주식 데이터를 가져와 패턴 분석을 수행하고 결과를 시각화합니다.
    
    Args:
        ticker (str): 주식 티커 심볼
        start_date (str): 시작 날짜 (YYYY-MM-DD)
        end_date (str): 종료 날짜 (YYYY-MM-DD)
        investment_style (str): 투자 스타일 ('long_term', 'swing_trade', 'default')
    """
    logging.info(f"{ticker} 데이터를 가져오는 중...")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        logging.error("데이터를 가져오지 못했습니다.")
        return
    
    logging.info(f"데이터 가져오기 완료: {len(data)} 행")
    
    # 패턴 분석기 초기화
    logging.info(f"패턴 분석 시작 (투자 스타일: {investment_style})...")
    analyzer = PatternAnalyzer(data)
    
    # 패턴 추가
    analyzer.add_pattern(BollingerBands(data))
    analyzer.add_pattern(DoubleBottom(data))
    
    # 투자 스타일에 맞는 신호 생성
    signals = analyzer.get_investment_style_signals(style=investment_style)
    
    # 결과 시각화
    visualize_results(data, signals, ticker, investment_style)
    
    # 백테스트
    backtest_results = backtest_strategy(data, signals)
    
    logging.info(f"백테스트 결과:")
    logging.info(f"  총 수익률: {backtest_results['total_return']:.2f}%")
    logging.info(f"  최대 낙폭: {backtest_results['max_drawdown']:.2f}%")
    logging.info(f"  승률: {backtest_results['win_rate']:.2f}%")
    logging.info(f"  매매 횟수: {backtest_results['trade_count']}")


def visualize_results(data: pd.DataFrame, signals: pd.DataFrame, ticker: str, investment_style: str) -> None:
    """
    분석 결과를 시각화합니다.
    
    Args:
        data (pd.DataFrame): 주식 데이터
        signals (pd.DataFrame): 매매 신호 데이터
        ticker (str): 주식 티커 심볼
        investment_style (str): 투자 스타일
    """
    plt.figure(figsize=(14, 10))
    
    # 주가 차트 (캔들스틱)
    ax1 = plt.subplot(2, 1, 1)
    
    # 그래프 설정
    plt.title(f"{ticker} 패턴 분석 결과 (투자 스타일: {investment_style})")
    plt.grid(True, alpha=0.3)
    
    # 캔들스틱 차트 대신 선 차트로 간소화
    plt.plot(data.index, data['Close'], color='black', label='Close Price')
    
    # 매수 신호 표시
    buy_signals = signals[signals['Signal'] == 1].index
    sell_signals = signals[signals['Signal'] == -1].index
    
    plt.scatter(buy_signals, data.loc[buy_signals, 'Close'], marker='^', color='g', s=100, label='Buy Signal')
    plt.scatter(sell_signals, data.loc[sell_signals, 'Close'], marker='v', color='r', s=100, label='Sell Signal')
    
    plt.legend()
    
    # 신호 차트
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(signals.index, signals['Combined'], color='blue', label='Combined Signal')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Buy Threshold')
    plt.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7, label='Sell Threshold')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.fill_between(signals.index, signals['Combined'], 0, where=signals['Combined'] >= 0, 
                    color='green', alpha=0.3)
    plt.fill_between(signals.index, signals['Combined'], 0, where=signals['Combined'] < 0, 
                    color='red', alpha=0.3)
    
    plt.title('신호 강도')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 그래프 저장
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f"{ticker}_{investment_style}_{timestamp}.png"))
    logging.info(f"차트 저장됨: {os.path.join(output_dir, f'{ticker}_{investment_style}_{timestamp}.png')}")
    
    plt.tight_layout()
    plt.show()


def backtest_strategy(data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    트레이딩 전략의 백테스트를 수행합니다.
    
    Args:
        data (pd.DataFrame): 주식 데이터
        signals (pd.DataFrame): 매매 신호 데이터
        initial_capital (float): 초기 자본
    
    Returns:
        dict: 백테스트 결과
    """
    # 포트폴리오 초기화
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['Position'] = 0  # 보유 포지션
    portfolio['Close'] = data['Close']
    portfolio['Cash'] = initial_capital
    portfolio['Holdings'] = 0.0
    portfolio['Total'] = portfolio['Cash']
    
    # 거래 내역
    trades = []
    
    # 포지션 상태 (0: 현금, 1: 주식)
    position = 0
    
    # 백테스트 실행
    for i in range(1, len(portfolio)):
        # 전일 포지션 유지
        portfolio.loc[portfolio.index[i], 'Position'] = portfolio.loc[portfolio.index[i-1], 'Position']
        portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash']
        
        # 매수 신호
        if signals.loc[portfolio.index[i], 'Signal'] == 1 and position == 0:
            # 전액 매수
            portfolio.loc[portfolio.index[i], 'Position'] = portfolio.loc[portfolio.index[i], 'Cash'] / portfolio.loc[portfolio.index[i], 'Close']
            portfolio.loc[portfolio.index[i], 'Cash'] = 0
            position = 1
            
            # 거래 기록
            trades.append({
                'date': portfolio.index[i],
                'type': 'buy',
                'price': portfolio.loc[portfolio.index[i], 'Close'],
                'shares': portfolio.loc[portfolio.index[i], 'Position']
            })
            
        # 매도 신호
        elif signals.loc[portfolio.index[i], 'Signal'] == -1 and position == 1:
            # 전량 매도
            portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Position'] * portfolio.loc[portfolio.index[i], 'Close']
            portfolio.loc[portfolio.index[i], 'Position'] = 0
            position = 0
            
            # 거래 기록
            trades.append({
                'date': portfolio.index[i],
                'type': 'sell',
                'price': portfolio.loc[portfolio.index[i], 'Close'],
                'shares': portfolio.loc[portfolio.index[i-1], 'Position']
            })
        
        # 보유 평가액 계산
        portfolio.loc[portfolio.index[i], 'Holdings'] = portfolio.loc[portfolio.index[i], 'Position'] * portfolio.loc[portfolio.index[i], 'Close']
        
        # 총 평가액 계산
        portfolio.loc[portfolio.index[i], 'Total'] = portfolio.loc[portfolio.index[i], 'Cash'] + portfolio.loc[portfolio.index[i], 'Holdings']
    
    # 수익률 계산
    total_return = (portfolio['Total'].iloc[-1] / initial_capital - 1) * 100
    
    # 최대 낙폭 계산
    cumulative_max = portfolio['Total'].cummax()
    drawdown = (portfolio['Total'] - cumulative_max) / cumulative_max * 100
    max_drawdown = drawdown.min()
    
    # 승률 계산
    win_count = 0
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            if trades[i+1]['price'] > trades[i]['price']:
                win_count += 1
    
    win_rate = (win_count / (len(trades) // 2)) * 100 if len(trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': len(trades) // 2,
        'portfolio': portfolio,
        'trades': trades
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='패턴 기반 주식 트레이딩 분석')
    parser.add_argument('--ticker', type=str, default='AAPL', help='주식 티커 심볼')
    parser.add_argument('--start', type=str, default='2022-01-01', help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--style', type=str, default='long_term', 
                        choices=['long_term', 'swing_trade', 'default'],
                        help='투자 스타일')
    
    args = parser.parse_args()
    
    run_pattern_analysis(args.ticker, args.start, args.end, args.style) 