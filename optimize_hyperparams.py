"""
하이퍼파라미터 최적화 스크립트

Optuna를 사용하여 강화학습 모델의 최적 하이퍼파라미터를 찾습니다.
"""

import optuna
import pandas as pd
import numpy as np
import torch
import os
import json
import logging
import argparse
from datetime import datetime
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# 프로젝트 모듈
from data.feature_engineering import create_enhanced_dataset
from reinforcement_learning.environment_enhanced import EnhancedStockTradingEnvironment
from reinforcement_learning.model import DQNAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 투자 스타일 기본 설정
INVESTMENT_STYLES = {
    "long_term": {
        "gamma_range": (0.95, 0.99),
        "transaction_cost_range": (0.001, 0.003),
        "holding_incentive_range": (0.0005, 0.002),
        "description": "장기 투자"
    },
    "swing_trade": {
        "gamma_range": (0.90, 0.97),
        "transaction_cost_range": (0.0005, 0.002),
        "holding_incentive_range": (0.0, 0.001),
        "description": "스윙 트레이딩"
    },
    "default": {
        "gamma_range": (0.90, 0.98),
        "transaction_cost_range": (0.0001, 0.001),
        "holding_incentive_range": (0.0, 0.0005),
        "description": "기본 스타일"
    }
}


def objective(trial, data, investment_style='default', quick_mode=False):
    """
    Optuna objective function

    Args:
        trial: Optuna trial 객체
        data: 학습 데이터
        investment_style: 투자 스타일
        quick_mode: 빠른 평가 모드 (에포크 감소)
    """
    style_config = INVESTMENT_STYLES.get(investment_style, INVESTMENT_STYLES['default'])

    # 하이퍼파라미터 샘플링
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', *style_config['gamma_range'])
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.990, 0.9999)

    # 네트워크 구조
    fc1_units = trial.suggest_categorical('fc1_units', [64, 128, 256])
    fc2_units = trial.suggest_categorical('fc2_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # 학습 설정
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    memory_size = trial.suggest_categorical('memory_size', [5000, 10000, 15000, 20000])

    # 환경 파라미터
    transaction_cost = trial.suggest_float('transaction_cost',
                                          *style_config['transaction_cost_range'])
    holding_incentive = trial.suggest_float('holding_incentive',
                                           *style_config['holding_incentive_range'])

    # 환경 생성
    env = EnhancedStockTradingEnvironment(
        data=data,
        initial_balance=100000,
        transaction_cost=transaction_cost,
        holding_incentive=holding_incentive,
        use_technical_indicators=True
    )

    state_size = env.get_state_dimension()
    action_size = 3

    # 에이전트 생성
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        fc1_units=fc1_units,
        fc2_units=fc2_units
    )

    # 학습 (축약)
    epochs = 20 if quick_mode else 30
    episode_rewards = []
    portfolio_returns = []

    for episode in range(epochs):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            state = next_state
            total_reward += reward

        # 포트폴리오 수익률 계산
        final_value = env.balance + env.shares_held * data['Close'].iloc[env.current_step]
        portfolio_return = (final_value - env.initial_balance) / env.initial_balance

        episode_rewards.append(total_reward)
        portfolio_returns.append(portfolio_return)

        # 중간 보고 (Pruning용)
        if episode % 5 == 0:
            avg_return = np.mean(portfolio_returns[-5:])
            trial.report(avg_return, episode)

            # 조기 종료 체크
            if trial.should_prune():
                raise optuna.TrialPruned()

    # 최종 성능: 샤프 비율 계산
    returns = np.array(portfolio_returns)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return sharpe_ratio


def optimize_hyperparameters(ticker='AAPL',
                             investment_style='default',
                             n_trials=50,
                             quick_mode=False):
    """
    하이퍼파라미터 최적화 실행

    Args:
        ticker: 주식 티커
        investment_style: 투자 스타일
        n_trials: 시도 횟수
        quick_mode: 빠른 모드 (각 trial의 에포크 감소)
    """
    print("=" * 80)
    print("🔧 Hyperparameter Optimization")
    print("=" * 80)
    print()
    print(f"Ticker: {ticker}")
    print(f"Investment Style: {investment_style}")
    print(f"Number of Trials: {n_trials}")
    print(f"Quick Mode: {quick_mode}")
    print()

    # 데이터 로드
    input_file = f'data/cleaned_{ticker}.csv'
    enhanced_file = input_file.replace('.csv', '_enhanced.csv')

    if not os.path.exists(enhanced_file):
        print(f"Creating enhanced dataset...")
        enhanced_data = create_enhanced_dataset(input_file, enhanced_file)
    else:
        print(f"Loading enhanced dataset: {enhanced_file}")
        enhanced_data = pd.read_csv(enhanced_file, index_col=0, parse_dates=True)

    print(f"Data loaded: {len(enhanced_data)} records")
    print()

    # Train/Validation 분할 (80/20)
    split_idx = int(len(enhanced_data) * 0.8)
    train_data = enhanced_data.iloc[:split_idx].copy()

    print(f"Training data: {len(train_data)} records")
    print()

    # Optuna Study 생성
    study_name = f"{ticker}_{investment_style}_optimization"
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    print("🚀 Starting optimization...")
    print()

    # 최적화 실행
    study.optimize(
        lambda trial: objective(trial, train_data, investment_style, quick_mode),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print()
    print("=" * 80)
    print("✅ Optimization Complete!")
    print("=" * 80)
    print()

    # 최적 파라미터
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial.number

    print(f"Best Trial: #{best_trial}")
    print(f"Best Sharpe Ratio: {best_value:.4f}")
    print()
    print("Best Hyperparameters:")
    print("-" * 80)
    for key, value in best_params.items():
        print(f"  {key:20s}: {value}")
    print()

    # 파라미터 중요도 계산
    try:
        importance = optuna.importance.get_param_importances(study)
        print("Parameter Importance:")
        print("-" * 80)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param:20s}: {imp:.4f}")
        print()
    except:
        importance = {}

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "ticker": ticker,
        "investment_style": investment_style,
        "optimization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": n_trials,
        "best_trial": best_trial,
        "best_value": float(best_value),
        "best_params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in best_params.items()},
        "parameter_importance": {k: float(v) for k, v in importance.items()}
    }

    config_file = f"{results_dir}/optimized_{ticker}_{investment_style}_config_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved: {config_file}")
    print()

    # 최적화 히스토리 저장
    df_trials = study.trials_dataframe()
    history_file = f"{results_dir}/optimization_history_{ticker}_{investment_style}_{timestamp}.csv"
    df_trials.to_csv(history_file, index=False)
    print(f"💾 History saved: {history_file}")
    print()

    print("=" * 80)
    print("💡 Next Steps")
    print("=" * 80)
    print()
    print("1. Train model with optimized parameters:")
    print(f"   python train_enhanced_model.py --ticker {ticker} --style {investment_style} \\")
    print(f"     --config {config_file} --epochs 100")
    print()
    print("2. Or manually specify the best parameters in your training script")
    print()

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize hyperparameters using Optuna')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--style', type=str, default='default',
                       choices=['default', 'long_term', 'swing_trade'],
                       help='Investment style')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs per trial)')

    args = parser.parse_args()

    config = optimize_hyperparameters(
        ticker=args.ticker,
        investment_style=args.style,
        n_trials=args.n_trials,
        quick_mode=args.quick
    )
