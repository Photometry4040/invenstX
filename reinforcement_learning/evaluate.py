# reinforcement_learning/evaluate.py

import numpy as np
import pandas as pd
import torch
import os
import logging
import sys
from datetime import datetime

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reinforcement_learning.model import DQNAgent
from reinforcement_learning.environment import StockTradingEnvironment
# 투자 스타일 설정 임포트
from reinforcement_learning.train import INVESTMENT_STYLES

def evaluate_agent(data, ticker="unknown", investment_style="default", seed=42):
    """
    학습된 에이전트를 불러와 주어진 데이터에서 평가합니다.
    
    Args:
        data (pd.DataFrame): 주식 데이터 (OHLCV 포함)
        ticker (str): 주식 티커 심볼
        investment_style (str): 투자 스타일 ('long_term', 'swing_trade', 'default')
        seed (int): 랜덤 시드
        
    Returns:
        dict: 평가 결과 (수익률, 거래 횟수 등)
    """
    # 랜덤 시드 고정
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 모델 파일 리스트 - 존재 여부 확인
    model_paths = [
        f"models/{ticker}_{investment_style}_model.pth",  # 투자 스타일별 모델
        f"models/{ticker}_model.pth",                     # 기본 모델
        f"models/{ticker}_default_model.pth",             # 기본 스타일 모델
        "dqn_model.pth",
        "best_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    # 모델 파일 존재 여부 확인
    if model_path is None:
        print(f"Error: No model file found for {ticker} with {investment_style} style")
        return None
    
    print(f"Using model: {model_path}")

    # 투자 스타일 설정 확인
    if investment_style not in INVESTMENT_STYLES:
        print(f"Warning: Investment style '{investment_style}' not found. Using default style.")
        investment_style = "default"
    
    style_config = INVESTMENT_STYLES[investment_style]
    print(f"Using investment style: {style_config['description']}")

    # Environment & Agent 초기화 - 투자 스타일 파라미터 추가
    state_size = 3
    action_size = 3
    env = StockTradingEnvironment(
        data,
        transaction_cost=style_config['transaction_cost'],
        holding_incentive=style_config['holding_incentive']
    )
    agent = DQNAgent(
        state_size, 
        action_size,
        gamma=style_config['gamma'],
        epsilon=0.0  # 평가 모드에서는 무작위성 제거
    )
    
    # 모델 로드 시도
    try:
        agent.load(model_path)
        logging.info(f"모델 로드 성공: {model_path}")
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"모델 로드 오류: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return None
    
    # 평가 실행
    return run_evaluation(env, agent, data)

def run_evaluation(env, agent, data):
    """
    강화학습 에이전트를 평가하고 결과를 반환합니다.
    
    Args:
        env: 환경 객체
        agent: 에이전트 객체
        data: 주식 데이터
        
    Returns:
        dict: 평가 결과
    """
    # 시뮬레이션
    state = env.reset()
    trades = []
    portfolio_values = []
    history = []
    
    # 데이터 인덱스 처리
    data.index = pd.to_datetime(data.index).normalize()
    
    # 초기값 설정
    initial_price = float(data['Close'].iloc[0].item())  # .item() 사용
    initial_portfolio = float(env.initial_balance)
    portfolio_values.append(initial_portfolio)
    
    step_count = 0
    
    while not env.done:
        # 평가 모드로 액션 선택 (무작위성 제거)
        action = agent.act(state, evaluation=True)
        next_state, reward, done = env.step(action)
        
        # 현재 날짜와 가격
        current_date = data.index[env.current_step]
        current_price = float(data['Close'].iloc[env.current_step].item())  # .item() 사용
        
        portfolio_value = env.balance + env.shares_held * current_price
        
        # history 기록
        history.append({
            "date": current_date,
            "portfolio_value": float(portfolio_value),
            "price": current_price
        })
        
        if action != 0:  # Buy/Sell 액션만 기록
            # 수익률 계산
            pct_change = ((portfolio_value - initial_portfolio) / initial_portfolio) * 100
            
            trades.append({
                "date": current_date,
                "price": current_price,
                "action": ["Hold", "Buy", "Sell"][action],
                "portfolio_value": float(portfolio_value),
                "total_return": float(pct_change),
                "balance": float(env.balance),
                "shares_held": float(env.shares_held)
            })
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: Portfolio value = ${float(portfolio_value):,.2f}")
        
        state = next_state
        portfolio_values.append(float(portfolio_value))  # float로 변환
        step_count += 1

    # 최종 포트폴리오 가치
    final_portfolio = float(portfolio_values[-1])
    total_return = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100

    # Buy & Hold 수익률 계산
    final_price = float(data['Close'].iloc[-1].item())  # .item() 사용
    bh_return = ((final_price - initial_price) / initial_price) * 100

    # 일간 수익률 계산
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        prev_val = portfolio_values[i-1]
        if prev_val != 0:
            daily_return = (portfolio_values[i] - prev_val) / prev_val
        else:
            daily_return = 0
        daily_returns.append(daily_return)

    # 샤프 비율 (연간화)
    if len(daily_returns) > 1 and np.std(daily_returns) != 0:
        sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))  # float로 변환
    else:
        sharpe_ratio = 0.0
    
    # 결과 딕셔너리 (모든 숫자 값을 float로 변환)
    results = {
        "Initial Balance": float(initial_portfolio), # 초기 자산
        "Final Balance": float(final_portfolio), # 최종 자산
        "Total Return (%)": float(total_return), # 총 수익률
        "Buy and Hold Return (%)": float(bh_return), # 주가 변동 없이 매일 매수/매도 했을 때의 수익률
        "Sharpe Ratio": float(sharpe_ratio), # 샤프 비율
        "Number of Trades": len(trades), # 거래 횟수
        "trades": trades, # 거래 기록
        "history": history, # 포트폴리오 가치 변동 기록
        "portfolio_values": portfolio_values # 포트폴리오 가치 배열 (일일 가치 추이)
    }
    
    print(f"Total trades recorded: {len(trades)}")
    return results

if __name__ == "__main__":
    # 디버깅 용
    from data.fetch_data import fetch_stock_data
    data = fetch_stock_data("AAPL", "2020-01-01", "2023-12-31")
    results = evaluate_agent(data, "AAPL")
    print(results)