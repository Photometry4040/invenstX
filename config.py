import os

# 모델 저장 경로
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# 데이터 설정
TICKER = "AAPL"
TRAIN_START_DATE = "2022-01-01"
TRAIN_END_DATE = "2023-12-31"
EVAL_START_DATE = "2024-10-01"
EVAL_END_DATE = "2024-12-31"

# 기술적 지표 설정
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 학습 설정
EPISODES = 100  # 에피소드 수
BATCH_SIZE = 64  # 배치 크기
MEMORY_SIZE = 5000  # 메모리 크기
GAMMA = 0.99  # 감가율
EPSILON = 1.0  # 초기 탐험률
EPSILON_MIN = 0.05  # 최소 탐험률
EPSILON_DECAY = 0.995  # 탐험률 감소율
LEARNING_RATE = 0.0001  # 학습률

# 환경 설정
INITIAL_BALANCE = 100000

#config.py: 설정값 관리
#main.py: 프로그램 진입점
#data/fetch_data.py: 주식 데이터 수집
#indicators/rsi.py: RSI 계산
#indicators/macd.py: MACD 계산
#reinforcement_learning/environment.py: 주식 거래 환경
#reinforcement_learning/model.py: DQN 모델 및 에이전트
#reinforcement_learning/train.py: 모델 훈련
#reinforcement_learning/evaluate.py: 모델 평가
#requirements.txt: 의존성 목록
#models/: 모델 저장 디렉토리 (자동 생성)