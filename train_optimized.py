"""
최적화된 하이퍼파라미터로 모델 학습

Optuna로 찾은 최적 파라미터를 사용하여 100 에포크 학습
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from data.feature_engineering import create_enhanced_dataset
from reinforcement_learning.environment_enhanced import EnhancedStockTradingEnvironment
from reinforcement_learning.model import DQNAgent

# 최적화된 설정 로드
config_file = 'results/optimized_AAPL_default_config_20251030_230331.json'
with open(config_file, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("🚀 Training with Optimized Hyperparameters")
print("=" * 80)
print()
print(f"Configuration: {config_file}")
print(f"Best Trial: #{config['best_trial']}")
print(f"Best Sharpe Ratio: {config['best_value']:.2f}")
print()

# 최적 파라미터
params = config['best_params']
print("Optimized Parameters:")
print("-" * 80)
for key, value in params.items():
    print(f"  {key:20s}: {value}")
print()

# 데이터 로드
ticker = config['ticker']
input_file = f'data/cleaned_{ticker}.csv'
enhanced_file = input_file.replace('.csv', '_enhanced.csv')

if not os.path.exists(enhanced_file):
    print(f"Creating enhanced dataset...")
    enhanced_data = create_enhanced_dataset(input_file, enhanced_file)
else:
    print(f"Loading enhanced dataset: {enhanced_file}")
    enhanced_data = pd.read_csv(enhanced_file, index_col=0, parse_dates=True)

print(f"✅ Data loaded: {len(enhanced_data)} records")
print()

# 환경 생성
print("Creating Environment...")
print("-" * 80)
env = EnhancedStockTradingEnvironment(
    data=enhanced_data,
    initial_balance=100000,
    transaction_cost=params['transaction_cost'],
    holding_incentive=params['holding_incentive'],
    use_technical_indicators=True
)

state_size = env.get_state_dimension()
action_size = 3

print(f"✅ Environment created")
print(f"   State dimension: {state_size}")
print(f"   Action space: {action_size}")
print()

# 에이전트 생성
print("Creating DQN Agent with Optimized Architecture...")
print("-" * 80)
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=params['learning_rate'],
    gamma=params['gamma'],
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=params['epsilon_decay'],
    memory_size=int(params['memory_size']),
    fc1_units=int(params['fc1_units']),
    fc2_units=int(params['fc2_units'])
)

print(f"✅ Agent created")
print(f"   Network: {state_size} → {int(params['fc1_units'])} → {int(params['fc2_units'])} → 64 → {action_size}")
print(f"   Learning rate: {params['learning_rate']:.6f}")
print(f"   Gamma: {params['gamma']:.4f}")
print(f"   Dropout: {params['dropout_rate']:.2f}")
print()

# 학습
epochs = 100
batch_size = int(params['batch_size'])

print("=" * 80)
print(f"🎯 Training: {epochs} Epochs")
print("=" * 80)
print()

best_reward = -np.inf
best_portfolio = 0
episode_rewards = []
episode_returns = []

for episode in range(epochs):
    state = env.reset()
    total_reward = 0
    steps = 0

    while not env.done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        state = next_state
        total_reward += reward
        steps += 1

    # 포트폴리오 수익률
    final_value = env.balance + env.shares_held * enhanced_data['Close'].iloc[env.current_step]
    portfolio_return = ((final_value - env.initial_balance) / env.initial_balance) * 100

    episode_rewards.append(total_reward)
    episode_returns.append(portfolio_return)

    # 최고 성과 업데이트
    if portfolio_return > best_portfolio:
        best_portfolio = portfolio_return
        best_reward = total_reward
        # 모델 저장
        model_path = f'models/{ticker}_default_optimized_model.pth'
        agent.save(model_path)

    # 10 에피소드마다 출력
    if (episode + 1) % 10 == 0:
        avg_reward_10 = np.mean(episode_rewards[-10:])
        avg_return_10 = np.mean(episode_returns[-10:])
        print(f"Episode {episode + 1}/{epochs}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Reward (last 10): {avg_reward_10:.2f}")
        print(f"  Portfolio Value: ${final_value:,.2f}")
        print(f"  Portfolio Return: {portfolio_return:.2f}%")
        print(f"  Avg Return (last 10): {avg_return_10:.2f}%")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print(f"  Steps: {steps}")
        print()

print("=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print()
print(f"Best Portfolio Return: {best_portfolio:.2f}%")
print(f"Best Total Reward: {best_reward:.2f}")
print(f"Model saved: models/{ticker}_default_optimized_model.pth")
print()

# 학습 기록 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_df = pd.DataFrame({
    'episode': range(1, epochs + 1),
    'reward': episode_rewards,
    'return': episode_returns
})
history_file = f"results/training_history_{ticker}_optimized_{timestamp}.csv"
history_df.to_csv(history_file, index=False)
print(f"💾 Training history saved: {history_file}")
print()
