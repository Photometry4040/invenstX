"""
빠른 RL 모델 학습
Paper Trading을 위한 23D 모델 학습
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch

from reinforcement_learning.model import DQNAgent
from reinforcement_learning.environment import StockTradingEnvironment

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    close = df['Close'].values
    volume = df['Volume'].values

    features_df = df.copy()

    # SMA
    features_df['SMA_5'] = df['Close'].rolling(5).mean()
    features_df['SMA_10'] = df['Close'].rolling(10).mean()
    features_df['SMA_20'] = df['Close'].rolling(20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features_df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    std_20 = df['Close'].rolling(20).std()
    features_df['BB_upper'] = features_df['SMA_20'] + 2 * std_20
    features_df['BB_lower'] = features_df['SMA_20'] - 2 * std_20

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    features_df['MACD'] = ema_12 - ema_26
    features_df['MACD_signal'] = features_df['MACD'].ewm(span=9).mean()

    # Volatility
    returns = df['Close'].pct_change()
    features_df['Volatility'] = returns.rolling(20).std()

    # Momentum
    features_df['Momentum_5'] = df['Close'].pct_change(5)
    features_df['Momentum_10'] = df['Close'].pct_change(10)

    # Volume
    features_df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    features_df['Volume_Ratio'] = df['Volume'] / features_df['Volume_SMA']

    # Price Range
    features_df['High_Low_Ratio'] = df['High'] / df['Low']

    # Fill NaN
    features_df = features_df.fillna(method='bfill').fillna(0)

    return features_df


def prepare_training_data(symbol='AAPL', days=365):
    """학습 데이터 준비"""
    logger.info(f"Loading {days} days of data for {symbol}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data for {symbol}")

    logger.info(f"Loaded {len(df)} days of data")

    # 기술적 지표 추가
    df_features = calculate_technical_indicators(df)

    return df_features


def train_quick_model(symbol='AAPL', episodes=30):
    """빠른 모델 학습"""
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Quick RL Model Training for {symbol}")
    logger.info(f"{'='*70}\n")

    # 데이터 준비
    data = prepare_training_data(symbol, days=365)
    logger.info(f"Data shape: {data.shape}")

    # 환경 생성
    env = StockTradingEnvironment(
        data=data,
        initial_balance=100000,
        transaction_cost=0.001
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f"State size: {state_size}")
    logger.info(f"Action size: {action_size}")

    # 에이전트 생성
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        fc1_units=128,
        fc2_units=128
    )

    # 학습
    logger.info(f"\n📚 Training for {episodes} episodes...")
    best_reward = -float('inf')
    best_return = -float('inf')

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # 학습
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        # 포트폴리오 수익률 계산
        portfolio_return = (env.portfolio_value - env.initial_balance) / env.initial_balance

        if total_reward > best_reward:
            best_reward = total_reward

        if portfolio_return > best_return:
            best_return = portfolio_return

        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode+1}/{episodes} | "
                       f"Reward: {total_reward:,.0f} | "
                       f"Return: {portfolio_return:+.2%} | "
                       f"Epsilon: {agent.epsilon:.3f} | "
                       f"Steps: {steps}")

    logger.info(f"\n✅ Training completed!")
    logger.info(f"Best reward: {best_reward:,.0f}")
    logger.info(f"Best return: {best_return:+.2%}")

    # 모델 저장
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"{symbol}_paper_trading_model.pth"
    torch.save(agent.model.state_dict(), model_path)
    logger.info(f"\n💾 Model saved to: {model_path}")

    return agent, env


def main():
    """메인 실행"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    print("\n" + "="*70)
    print("🎯 Quick RL Model Training for Paper Trading")
    print("="*70)
    print(f"\nSymbols to train: {', '.join(symbols)}")
    print(f"Episodes per symbol: 30 (quick training)")

    response = input("\n학습을 시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ 취소되었습니다.")
        return

    for symbol in symbols:
        try:
            agent, env = train_quick_model(symbol, episodes=30)
            print(f"\n✅ {symbol} training completed!\n")
        except Exception as e:
            logger.error(f"\n❌ Error training {symbol}: {e}\n")

    print("\n" + "="*70)
    print("✅ All models trained successfully!")
    print("="*70)
    print("\n다음 단계: python rl_paper_trading.py")


if __name__ == "__main__":
    main()