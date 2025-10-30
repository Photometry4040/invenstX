"""
Enhanced 모델 학습 스크립트

확장된 State representation을 사용하여 모델을 학습합니다.
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
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


def train_enhanced_model(ticker: str = 'AAPL',
                        input_file: str = 'data/cleaned_AAPL.csv',
                        epochs: int = 50,
                        batch_size: int = 32,
                        investment_style: str = 'default'):
    """
    확장된 피처로 모델 학습

    Args:
        ticker: 주식 티커
        input_file: 정제된 데이터 파일
        epochs: 학습 에포크
        batch_size: 배치 크기
        investment_style: 투자 스타일
    """
    print("=" * 80)
    print("🚀 Enhanced Model Training")
    print("=" * 80)
    print()

    # 투자 스타일 설정
    INVESTMENT_STYLES = {
        "long_term": {
            "gamma": 0.99,
            "transaction_cost": 0.002,
            "holding_incentive": 0.001
        },
        "swing_trade": {
            "gamma": 0.95,
            "transaction_cost": 0.001,
            "holding_incentive": 0.0005
        },
        "default": {
            "gamma": 0.95,
            "transaction_cost": 0.0005,
            "holding_incentive": 0.0
        }
    }

    style_config = INVESTMENT_STYLES.get(investment_style, INVESTMENT_STYLES['default'])

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
    print(f"   Investment style: {investment_style}")
    print()

    # Step 3: 에이전트 생성
    print("3️⃣  Creating DQN Agent")
    print("-" * 80)

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0001,
        gamma=style_config['gamma'],
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000
    )

    print(f"✅ Agent created")
    print(f"   Network: {state_size} → 128 → 128 → 64 → {action_size}")
    print(f"   Gamma: {style_config['gamma']}")
    print()

    # Step 4: 학습
    print("4️⃣  Training")
    print("-" * 80)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()

    best_reward = -np.inf
    best_portfolio = 0
    episode_rewards = []

    for episode in range(epochs):
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            # 행동 선택
            action = agent.act(state)

            # 행동 수행
            next_state, reward, done = env.step(action)

            # 경험 저장
            agent.remember(state, action, reward, next_state, done)

            # 학습
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            state = next_state
            total_reward += reward
            steps += 1

        # 에피소드 종료
        portfolio_value = env.get_portfolio_value()
        portfolio_return = (portfolio_value - env.initial_balance) / env.initial_balance * 100

        episode_rewards.append(total_reward)

        # 최고 성능 모델 저장
        if portfolio_return > best_portfolio:
            best_portfolio = portfolio_return
            best_reward = total_reward

            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/{ticker}_{investment_style}_enhanced_model.pth"
            agent.save(model_path)

        # 로깅
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{epochs}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Portfolio Value: ${portfolio_value:,.2f}")
            print(f"  Portfolio Return: {portfolio_return:.2f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Steps: {steps}")
            print()

    # Step 5: 결과 요약
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print()
    print(f"Best Portfolio Return: {best_portfolio:.2f}%")
    print(f"Best Total Reward: {best_reward:.2f}")
    print(f"Model saved: models/{ticker}_{investment_style}_enhanced_model.pth")
    print()

    # 학습 곡선 저장
    results_df = pd.DataFrame({
        'episode': range(1, epochs + 1),
        'reward': episode_rewards
    })
    results_file = f"results/training_history_{ticker}_{investment_style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"Training history saved: {results_file}")
    print()

    return agent, env


def compare_models(ticker: str = 'AAPL',
                  test_file: str = 'data/cleaned_AAPL.csv'):
    """
    기본 모델 vs 향상된 모델 성능 비교
    """
    print("=" * 80)
    print("📊 Model Comparison: Basic vs Enhanced")
    print("=" * 80)
    print()

    # TODO: 구현 예정
    print("⚠️  Comparison feature coming soon!")
    print()
    print("비교 항목:")
    print("  - Total Return")
    print("  - Sharpe Ratio")
    print("  - Max Drawdown")
    print("  - Win Rate")
    print("  - Number of Trades")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train enhanced RL model with technical indicators')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--input', type=str, default='data/cleaned_AAPL.csv', help='Input data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--style', type=str, default='default',
                       choices=['default', 'long_term', 'swing_trade'],
                       help='Investment style')

    args = parser.parse_args()

    # 학습 실행
    agent, env = train_enhanced_model(
        ticker=args.ticker,
        input_file=args.input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        investment_style=args.style
    )

    print("=" * 80)
    print("💡 Next Steps")
    print("=" * 80)
    print()
    print("1. 모델 평가:")
    print("   python evaluate_enhanced_model.py --ticker AAPL --style default")
    print()
    print("2. 하이퍼파라미터 최적화:")
    print("   /optimize-hyperparams AAPL default 50")
    print()
    print("3. 성능 비교:")
    print("   /analyze-model AAPL default")
    print()
