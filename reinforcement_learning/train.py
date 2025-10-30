# reinforcement_learning/train.py
import numpy as np
import logging
from itertools import product
from .environment import StockTradingEnvironment
from .model import DQNAgent
import torch
from tqdm import tqdm
import time
import sys
import os

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BATCH_SIZE, EPISODES  # 절대 경로로 import

logging.basicConfig(level=logging.INFO)

# 투자 스타일 정의
INVESTMENT_STYLES = {
    "long_term": {
        "description": "장기 투자 (1년 이상 보유)",
        "gamma": 0.99,            # 미래 보상에 높은 가중치
        "reward_window": 100,     # 장기 수익률 평가
        "holding_incentive": 0.001, # 매수 후 홀딩에 대한 인센티브
        "transaction_cost": 0.002   # 거래 비용 (높게 설정하여 잦은 거래 제한)
    },
    "swing_trade": {
        "description": "중단기 스윙 트레이드 (1주-3개월 보유)",
        "gamma": 0.95,            # 중단기 미래 보상 가중치
        "reward_window": 20,      # 중단기 수익률 평가
        "holding_incentive": 0.0005, # 매수 후 적절한 기간 동안 홀딩에 대한 인센티브
        "transaction_cost": 0.001   # 적절한 거래 비용
    },
    "default": {
        "description": "기본 트레이딩 전략",
        "gamma": 0.95,            # 기본 감가율
        "reward_window": 5,       # 단기 수익률 평가
        "holding_incentive": 0.0,  # 홀딩 인센티브 없음
        "transaction_cost": 0.0005  # 낮은 거래 비용
    }
}

def train_agent(data, ticker, epochs=50, batch_size=32, investment_style="default"):
    """
    주어진 데이터로 에이전트를 훈련시키고, 
    훈련된 모델을 티커별로 저장
    
    Args:
        data (pd.DataFrame): 주식 데이터
        ticker (str): 주식 티커 심볼
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 사이즈
        investment_style (str): 투자 스타일 ('long_term', 'swing_trade', 'default')
    """
    # 선택한 투자 스타일 설정 적용
    if investment_style not in INVESTMENT_STYLES:
        logging.warning(f"지정한 투자 스타일 '{investment_style}'을 찾을 수 없습니다. 기본 스타일을 사용합니다.")
        investment_style = "default"
    
    style_config = INVESTMENT_STYLES[investment_style]
    logging.info(f"선택된 투자 스타일: {style_config['description']}")
    
    # 모델 저장 디렉토리 생성
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = f"{model_dir}/{ticker}_{investment_style}_model.pth"
    state_size = 3  # [price, balance, shares]
    action_size = 3  # [buy, sell, hold]
    best_reward = -np.inf
    best_params = None
    patience = 10  # 조기 종료를 위한 인내심
    no_improve = 0

    pbar = tqdm(total=epochs, desc="Training")
    start_time = time.time()

    # 투자 스타일에 맞게 환경 파라미터 설정
    env = StockTradingEnvironment(
        data, 
        transaction_cost=style_config['transaction_cost'],
        holding_incentive=style_config['holding_incentive']
    )
    
    # 투자 스타일에 맞게 에이전트 파라미터 설정
    agent = DQNAgent(
        state_size, 
        action_size, 
        gamma=style_config['gamma']
    )
    
    total_rewards = []
    avg_window = style_config['reward_window']  # 투자 스타일에 맞는 평가 윈도우

    for episode in range(epochs):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            # 배치 크기만큼 메모리가 쌓이면 학습 수행
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
                
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        avg_reward = (np.mean(total_rewards[-avg_window:])
                      if len(total_rewards) >= avg_window
                      else np.mean(total_rewards))

        # 성능 향상 체크 및 조기 종료
        if avg_reward > best_reward:
            best_reward = avg_reward
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.save(model_path)
            logging.info(f"최고 성능 모델 저장됨 (보상: {best_reward:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info("성능 향상이 없어 학습을 조기 종료합니다")
                break

        pbar.update(1)
        pbar.set_postfix({'reward': total_reward, 'avg_reward': avg_reward})

    training_time = time.time() - start_time
    logging.info(f"학습 완료 (소요 시간: {training_time:.2f}초)")
    pbar.close()

    return model_path

if __name__ == "__main__":
    from data.fetch_data import fetch_stock_data
    
    # 기본 예제 실행
    data = fetch_stock_data("AAPL", "2020-01-01", "2023-12-31")
    
    # 투자 스타일 선택 (명령줄 인수로 받거나 기본값 사용)
    import argparse
    parser = argparse.ArgumentParser(description='주식 거래 강화학습 모델 훈련')
    parser.add_argument('--style', type=str, default='default',
                        choices=INVESTMENT_STYLES.keys(),
                        help='투자 스타일 (long_term, swing_trade, default)')
    args = parser.parse_args()
    
    # 선택한 투자 스타일로 훈련
    train_agent(data, "AAPL", investment_style=args.style)