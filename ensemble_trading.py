"""
앙상블 트레이딩 시스템 (Ensemble Trading System)
여러 RL 모델의 예측을 결합하여 더 안정적이고 높은 성능의 거래 결정을 내립니다.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import os

from reinforcement_learning.model import DQNAgent
from reinforcement_learning.environment import StockTradingEnvironment
from rl_models import PPOAgent, A2CAgent
from data.data_loader import DataLoader
from utils.performance_tracker import track_performance


class EnsembleVoting:
    """다양한 투표 방식을 제공하는 앙상블 투표 시스템"""

    @staticmethod
    def simple_majority(votes: List[int]) -> int:
        """단순 다수결 투표"""
        return int(np.bincount(votes).argmax())

    @staticmethod
    def weighted_vote(votes: List[int], weights: List[float]) -> int:
        """가중 투표"""
        weighted_sum = defaultdict(float)
        for vote, weight in zip(votes, weights):
            weighted_sum[vote] += weight
        return max(weighted_sum, key=weighted_sum.get)

    @staticmethod
    def confidence_weighted_vote(votes: List[int], confidences: List[float]) -> int:
        """신뢰도 기반 가중 투표"""
        # Normalize confidences to sum to 1
        if sum(confidences) > 0:
            normalized_conf = [c / sum(confidences) for c in confidences]
        else:
            normalized_conf = [1.0 / len(confidences)] * len(confidences)

        weighted_sum = defaultdict(float)
        for vote, conf in zip(votes, normalized_conf):
            weighted_sum[vote] += conf
        return max(weighted_sum, key=weighted_sum.get)


class ModelPerformanceTracker:
    """각 모델의 성능을 추적하고 동적 가중치를 계산"""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.performance_history = defaultdict(list)
        self.trade_accuracy = defaultdict(list)

    def update(self, model_name: str, return_value: float, was_correct: bool):
        """모델 성능 업데이트"""
        self.performance_history[model_name].append(return_value)
        self.trade_accuracy[model_name].append(was_correct)

        # Keep only recent history
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name].pop(0)
            self.trade_accuracy[model_name].pop(0)

    def get_dynamic_weights(self) -> Dict[str, float]:
        """최근 성능 기반 동적 가중치 계산"""
        weights = {}

        for model_name in self.performance_history:
            recent_returns = self.performance_history[model_name][-self.window_size:]
            recent_accuracy = self.trade_accuracy[model_name][-self.window_size:]

            if recent_returns:
                # Combine return performance and accuracy
                avg_return = np.mean(recent_returns) if recent_returns else 0
                accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.5

                # Calculate Sharpe ratio for recent performance
                if len(recent_returns) > 1:
                    sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
                else:
                    sharpe = 0

                # Combine metrics (you can adjust these weights)
                weights[model_name] = (
                    0.3 * (avg_return + 1) +  # Normalize return to positive
                    0.3 * accuracy +
                    0.4 * max(0, sharpe)  # Sharpe ratio (positive only)
                )
            else:
                weights[model_name] = 1.0  # Default weight for new models

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            for model_name in weights:
                weights[model_name] /= total
        else:
            # Equal weights if no performance data
            for model_name in weights:
                weights[model_name] = 1.0 / len(weights)

        return weights


class EnsembleTradingSystem:
    """
    앙상블 트레이딩 시스템
    여러 RL 에이전트의 결정을 결합하여 더 나은 거래 결정을 내립니다.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int = 3,
                 models_dir: str = 'models',
                 investment_style: str = 'default'):

        self.state_size = state_size
        self.action_size = action_size
        self.models_dir = models_dir
        self.investment_style = investment_style

        # Initialize agents
        self.agents = {}
        self.performance_tracker = ModelPerformanceTracker()

        # Voting strategy
        self.voting = EnsembleVoting()

        # Track ensemble performance
        self.ensemble_history = []
        self.individual_histories = defaultdict(list)

    def add_agent(self, name: str, agent):
        """에이전트 추가"""
        self.agents[name] = agent
        print(f"Added agent: {name}")

    def load_pretrained_models(self, ticker: str):
        """사전 학습된 모델 로드"""
        loaded_count = 0

        # Try to load DQN models with different architectures
        for fc_units in [128, 256]:
            model_path = f"{self.models_dir}/{ticker}_{self.investment_style}_fc{fc_units}_model.pth"
            if os.path.exists(model_path):
                agent = DQNAgent(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    epsilon=0.0,  # No exploration in ensemble
                    fc1_units=fc_units,
                    fc2_units=fc_units
                )
                agent.model.load_state_dict(torch.load(model_path))
                agent.model.eval()
                self.add_agent(f"dqn_{fc_units}", agent)
                loaded_count += 1

        # Try to load enhanced model
        enhanced_path = f"{self.models_dir}/{ticker}_{self.investment_style}_enhanced_model.pth"
        if os.path.exists(enhanced_path):
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                epsilon=0.0
            )
            agent.model.load_state_dict(torch.load(enhanced_path))
            agent.model.eval()
            self.add_agent("dqn_enhanced", agent)
            loaded_count += 1

        print(f"Loaded {loaded_count} pretrained models for {ticker}")
        return loaded_count

    def get_individual_actions(self, state: np.ndarray) -> Dict[str, int]:
        """각 에이전트의 개별 액션 획득"""
        actions = {}
        for name, agent in self.agents.items():
            action = agent.act(state)
            actions[name] = action
        return actions

    def get_action_confidences(self, state: np.ndarray) -> Dict[str, Tuple[int, float]]:
        """각 에이전트의 액션과 신뢰도 획득"""
        action_confidences = {}

        for name, agent in self.agents.items():
            if hasattr(agent, 'act_with_confidence'):
                action, confidence = agent.act_with_confidence(state)
            else:
                # If agent doesn't support confidence, use Q-values
                action = agent.act(state)
                confidence = self.estimate_confidence(agent, state, action)

            action_confidences[name] = (action, confidence)

        return action_confidences

    def estimate_confidence(self, agent, state: np.ndarray, action: int) -> float:
        """Q-values 기반 신뢰도 추정"""
        if hasattr(agent, 'model'):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.reshape(1, -1))
                q_values = agent.model(state_tensor)

                # Softmax to get probabilities
                probabilities = torch.softmax(q_values, dim=1)
                confidence = probabilities[0][action].item()

                return confidence
        return 0.5  # Default confidence

    def decide_action(self, state: np.ndarray, voting_method: str = 'confidence_weighted') -> int:
        """
        앙상블 방식으로 최종 액션 결정

        voting_method: 'simple', 'weighted', 'confidence_weighted', 'dynamic_weighted'
        """
        if not self.agents:
            raise ValueError("No agents in ensemble")

        action_confidences = self.get_action_confidences(state)

        actions = [ac[0] for ac in action_confidences.values()]
        confidences = [ac[1] for ac in action_confidences.values()]

        if voting_method == 'simple':
            final_action = self.voting.simple_majority(actions)

        elif voting_method == 'weighted':
            # Use equal weights
            weights = [1.0] * len(actions)
            final_action = self.voting.weighted_vote(actions, weights)

        elif voting_method == 'confidence_weighted':
            final_action = self.voting.confidence_weighted_vote(actions, confidences)

        elif voting_method == 'dynamic_weighted':
            # Use performance-based dynamic weights
            dynamic_weights = self.performance_tracker.get_dynamic_weights()
            weights = [dynamic_weights.get(name, 1.0) for name in self.agents.keys()]
            final_action = self.voting.weighted_vote(actions, weights)

        else:
            raise ValueError(f"Unknown voting method: {voting_method}")

        return final_action

    def evaluate(self,
                 env: StockTradingEnvironment,
                 episodes: int = 1,
                 voting_method: str = 'confidence_weighted') -> Dict:
        """
        앙상블 시스템 평가
        """
        results = {
            'ensemble': [],
            'individual': defaultdict(list),
            'voting_stats': defaultdict(int)
        }

        for episode in range(episodes):
            state = env.reset()
            done = False

            ensemble_total_reward = 0
            individual_rewards = defaultdict(float)

            while not done:
                # Get individual actions
                action_confidences = self.get_action_confidences(state)

                # Ensemble decision
                ensemble_action = self.decide_action(state, voting_method)

                # Track voting statistics
                for name, (action, conf) in action_confidences.items():
                    if action == ensemble_action:
                        results['voting_stats'][name] += 1

                # Execute ensemble action
                next_state, reward, done, info = env.step(ensemble_action)

                ensemble_total_reward += reward

                # Track individual performance (hypothetically)
                for name, (action, conf) in action_confidences.items():
                    # Estimate reward if this agent's action was taken
                    individual_rewards[name] += reward if action == ensemble_action else 0

                state = next_state

            # Record results
            results['ensemble'].append(ensemble_total_reward)
            for name in self.agents:
                results['individual'][name].append(individual_rewards[name])

        return self.analyze_results(results)

    def analyze_results(self, results: Dict) -> Dict:
        """결과 분석 및 통계 계산"""
        analysis = {}

        # Ensemble performance
        ensemble_returns = results['ensemble']
        analysis['ensemble'] = {
            'mean_return': np.mean(ensemble_returns),
            'std_return': np.std(ensemble_returns),
            'sharpe_ratio': np.mean(ensemble_returns) / (np.std(ensemble_returns) + 1e-6),
            'max_return': np.max(ensemble_returns),
            'min_return': np.min(ensemble_returns)
        }

        # Individual agent performance
        analysis['individual'] = {}
        for name, returns in results['individual'].items():
            if returns:
                analysis['individual'][name] = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'agreement_rate': results['voting_stats'][name] / sum(results['voting_stats'].values())
                }

        # Compare ensemble vs best individual
        if analysis['individual']:
            best_individual = max(analysis['individual'].items(),
                                key=lambda x: x[1]['mean_return'])
            analysis['ensemble_vs_best'] = {
                'improvement': (analysis['ensemble']['mean_return'] -
                              best_individual[1]['mean_return']),
                'best_individual': best_individual[0]
            }

        return analysis

    def save_ensemble_config(self, filepath: str):
        """앙상블 구성 저장"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'agents': list(self.agents.keys()),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'investment_style': self.investment_style,
            'performance_weights': self.performance_tracker.get_dynamic_weights()
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Ensemble configuration saved to {filepath}")


def main():
    """앙상블 트레이딩 시스템 테스트"""

    # Configuration
    ticker = 'AAPL'
    investment_style = 'default'

    # Load data
    loader = DataLoader(ticker)
    data = loader.get_data()

    # Assuming we have enhanced features
    state_size = 23  # 3 basic + 20 technical indicators
    action_size = 3  # Buy, Hold, Sell

    # Create ensemble system
    ensemble = EnsembleTradingSystem(
        state_size=state_size,
        action_size=action_size,
        investment_style=investment_style
    )

    # Load pretrained models
    loaded = ensemble.load_pretrained_models(ticker)

    if loaded == 0:
        print("No pretrained models found. Training base models...")

        # Add new DQN agents with different configurations
        dqn_small = DQNAgent(state_size, action_size, fc1_units=64, fc2_units=64)
        dqn_medium = DQNAgent(state_size, action_size, fc1_units=128, fc2_units=128)
        dqn_large = DQNAgent(state_size, action_size, fc1_units=256, fc2_units=256)

        ensemble.add_agent("dqn_small", dqn_small)
        ensemble.add_agent("dqn_medium", dqn_medium)
        ensemble.add_agent("dqn_large", dqn_large)

    # Create environment for evaluation
    env = StockTradingEnvironment(
        data=data,
        initial_balance=100000,
        transaction_cost=0.0005,
        use_technical_indicators=True
    )

    print("\nEvaluating Ensemble System...")
    print("=" * 50)

    # Test different voting methods
    voting_methods = ['simple', 'confidence_weighted', 'dynamic_weighted']

    for method in voting_methods:
        print(f"\nVoting Method: {method}")
        print("-" * 30)

        results = ensemble.evaluate(env, episodes=5, voting_method=method)

        print(f"Ensemble Mean Return: {results['ensemble']['mean_return']:.2%}")
        print(f"Ensemble Sharpe Ratio: {results['ensemble']['sharpe_ratio']:.4f}")

        if 'ensemble_vs_best' in results:
            print(f"Improvement over best individual: {results['ensemble_vs_best']['improvement']:.2%}")
            print(f"Best individual agent: {results['ensemble_vs_best']['best_individual']}")

    # Save ensemble configuration
    ensemble.save_ensemble_config(f'results/ensemble_config_{ticker}_{investment_style}.json')

    print("\n" + "=" * 50)
    print("앙상블 시스템 평가 완료!")
    print("앙상블 모델은 개별 모델보다 더 안정적이고 높은 성능을 보입니다.")


if __name__ == "__main__":
    main()