"""
Enhanced 모델 평가 스크립트

23D 상태 공간을 사용하는 강화 모델을 평가합니다.
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
import argparse
from datetime import datetime

# 프로젝트 모듈 임포트
from data.feature_engineering import create_enhanced_dataset
from reinforcement_learning.environment_enhanced import EnhancedStockTradingEnvironment
from reinforcement_learning.model import DQNAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def evaluate_enhanced_model(ticker: str = 'AAPL',
                            input_file: str = None,
                            investment_style: str = 'default',
                            model_path: str = None):
    """
    확장된 피처를 사용하는 모델 평가

    Args:
        ticker: 주식 티커
        input_file: 정제된 데이터 파일
        investment_style: 투자 스타일
        model_path: 모델 파일 경로 (없으면 자동 탐색)
    """
    print("=" * 80)
    print("📊 Enhanced Model Evaluation")
    print("=" * 80)
    print()

    # 투자 스타일 설정
    INVESTMENT_STYLES = {
        "long_term": {
            "gamma": 0.99,
            "transaction_cost": 0.002,
            "holding_incentive": 0.001,
            "description": "장기 투자"
        },
        "swing_trade": {
            "gamma": 0.95,
            "transaction_cost": 0.001,
            "holding_incentive": 0.0005,
            "description": "스윙 트레이딩"
        },
        "default": {
            "gamma": 0.95,
            "transaction_cost": 0.0005,
            "holding_incentive": 0.0,
            "description": "기본 스타일"
        }
    }

    style_config = INVESTMENT_STYLES.get(investment_style, INVESTMENT_STYLES['default'])

    # 입력 파일 설정
    if input_file is None:
        input_file = f'data/cleaned_{ticker}.csv'

    # 모델 파일 찾기
    if model_path is None:
        # 가능한 모델 경로들
        possible_paths = [
            f'models/{ticker}_{investment_style}_enhanced_model.pth',
            f'models/{ticker}_default_enhanced_model.pth',
            f'models/{ticker}_enhanced_model.pth'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print(f"❌ Error: No enhanced model found for {ticker}")
            return None

    print(f"📁 Using model: {model_path}")
    print(f"📈 Investment style: {style_config['description']}")
    print()

    # Step 1: 피처 엔지니어링
    print("1️⃣  Feature Engineering")
    print("-" * 80)

    enhanced_file = input_file.replace('.csv', '_enhanced.csv')

    if not os.path.exists(enhanced_file):
        print(f"Creating enhanced dataset from {input_file}...")
        enhanced_data = create_enhanced_dataset(input_file, enhanced_file)
    else:
        print(f"Loading existing enhanced dataset: {enhanced_file}")
        enhanced_data = pd.read_csv(enhanced_file, index_col=0, parse_dates=True)

    print(f"✅ Enhanced data loaded: {len(enhanced_data)} records, {len(enhanced_data.columns)} features")
    print()

    # Step 2: 환경 생성
    print("2️⃣  Creating Enhanced Environment")
    print("-" * 80)

    env = EnhancedStockTradingEnvironment(
        data=enhanced_data,
        initial_balance=100000,
        transaction_cost=style_config['transaction_cost'],
        holding_incentive=style_config['holding_incentive'],
        use_technical_indicators=True
    )

    state_size = env.get_state_dimension()
    action_size = 3  # Hold, Buy, Sell

    print(f"✅ Environment created")
    print(f"   State dimension: {state_size}")
    print(f"   Action space: {action_size}")
    print()

    # Step 3: 에이전트 생성 및 모델 로드
    print("3️⃣  Loading DQN Agent")
    print("-" * 80)

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0001,
        gamma=style_config['gamma'],
        epsilon=0.0,  # 평가 모드: 무작위 행동 제거
        epsilon_min=0.0,
        epsilon_decay=1.0,
        memory_size=10000
    )

    try:
        agent.load(model_path)
        print(f"✅ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

    # Step 4: 평가 실행
    print("4️⃣  Running Evaluation")
    print("-" * 80)
    print()

    state = env.reset()
    trades = []
    portfolio_values = []
    history = []

    # 초기값 설정
    initial_price = float(enhanced_data['Close'].iloc[0])
    initial_portfolio = float(env.initial_balance)
    portfolio_values.append(initial_portfolio)

    step_count = 0

    while not env.done:
        # 평가 모드로 액션 선택 (무작위성 제거)
        action = agent.act(state, evaluation=True)
        next_state, reward, done = env.step(action)

        # 현재 날짜와 가격
        current_date = enhanced_data.index[env.current_step]
        current_price = float(enhanced_data['Close'].iloc[env.current_step])

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
        portfolio_values.append(float(portfolio_value))
        step_count += 1

    print()
    print("=" * 80)
    print("✅ Evaluation Complete!")
    print("=" * 80)
    print()

    # 최종 포트폴리오 가치
    final_portfolio = float(portfolio_values[-1])
    total_return = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100

    # Buy & Hold 수익률 계산
    final_price = float(enhanced_data['Close'].iloc[-1])
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
        sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # 결과 출력
    print("📊 Performance Metrics")
    print("-" * 80)
    print(f"Initial Balance:        ${initial_portfolio:,.2f}")
    print(f"Final Balance:          ${final_portfolio:,.2f}")
    print(f"Total Return:           {total_return:,.2f}%")
    print(f"Buy & Hold Return:      {bh_return:,.2f}%")
    print(f"Outperformance:         {total_return - bh_return:,.2f}%")
    print(f"Sharpe Ratio:           {sharpe_ratio:.4f}")
    print(f"Number of Trades:       {len(trades)}")
    print()

    # 결과 딕셔너리
    results = {
        "ticker": ticker,
        "investment_style": investment_style,
        "model_path": model_path,
        "initial_balance": float(initial_portfolio),
        "final_balance": float(final_portfolio),
        "total_return": float(total_return),
        "buy_hold_return": float(bh_return),
        "outperformance": float(total_return - bh_return),
        "sharpe_ratio": float(sharpe_ratio),
        "num_trades": len(trades),
        "trades": trades,
        "history": history,
        "portfolio_values": portfolio_values
    }

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 거래 기록 저장
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_file = f"{results_dir}/trades_{ticker}_{investment_style}_enhanced_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"💾 Trades saved: {trades_file}")

    # 포트폴리오 히스토리 저장
    history_df = pd.DataFrame(history)
    history_file = f"{results_dir}/portfolio_history_{ticker}_{investment_style}_enhanced_{timestamp}.csv"
    history_df.to_csv(history_file, index=False)
    print(f"💾 Portfolio history saved: {history_file}")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate enhanced RL model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--input-file', type=str, default=None, help='Input data file')
    parser.add_argument('--style', type=str, default='default',
                       choices=['default', 'long_term', 'swing_trade'],
                       help='Investment style')
    parser.add_argument('--model-path', type=str, default=None, help='Model file path')

    args = parser.parse_args()

    results = evaluate_enhanced_model(
        ticker=args.ticker,
        input_file=args.input_file,
        investment_style=args.style,
        model_path=args.model_path
    )

    if results is None:
        exit(1)
