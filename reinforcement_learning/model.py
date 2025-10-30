# reinforcement_learning/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import os
import logging

# M1 GPU 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
        """
        심층 Q 네트워크 모델 초기화
        
        Args:
            state_size (int): 상태 공간의 차원 (입력 크기)
            action_size (int): 행동 공간의 차원 (출력 크기)
            fc1_units (int): 첫 번째 은닉층의 유닛 수
            fc2_units (int): 두 번째 은닉층의 유닛 수
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)  # 배치 정규화 추가
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)  # 배치 정규화 추가
        self.fc3 = nn.Linear(fc2_units, 64)
        self.bn3 = nn.BatchNorm1d(64)  # 배치 정규화 추가
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
        self.to(device)

    def forward(self, x):
        """
        신경망 순전파
        
        Args:
            x (torch.Tensor): 입력 텐서
            
        Returns:
            torch.Tensor: 각 행동의 Q 값
        """
        # 배치 크기가 1인 경우 배치 정규화를 건너뜀
        if x.size(0) == 1:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                fc1_units=128, fc2_units=128):
        """
        DQN 에이전트 초기화

        Args:
            state_size (int): 상태 공간의 크기
            action_size (int): 행동 공간의 크기
            learning_rate (float): 학습률
            gamma (float): 할인율
            epsilon (float): 초기 탐험률
            epsilon_min (float): 최소 탐험률
            epsilon_decay (float): 탐험률 감소율
            memory_size (int): 리플레이 메모리 크기
            fc1_units (int): 첫 번째 은닉층의 유닛 수
            fc2_units (int): 두 번째 은닉층의 유닛 수
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.memory_size = memory_size
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # 탐험률
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 투자 스타일에 맞는 모델 확장
        self.model = DQN(state_size, action_size, fc1_units, fc2_units)
        self.target_model = DQN(state_size, action_size, fc1_units, fc2_units)  # 타겟 네트워크 추가
        self.update_target_model()  # 타겟 네트워크 초기화

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_counter = 0
        self.target_update_frequency = 10  # 10번의 학습마다 타겟 네트워크 업데이트

    def update_target_model(self):
        """타겟 네트워크 업데이트"""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, evaluation=False):
        """
        현재 상태에서 액션을 선택합니다.
        
        Args:
            state: 현재 상태
            evaluation: 평가 모드 여부 (True: 무작위성 제거)
            
        Returns:
            int: 선택된 액션 (0: Hold, 1: Buy, 2: Sell)
        """
        if evaluation:
            epsilon = 0.0  # 평가 시에는 무작위성 제거
        else:
            epsilon = self.epsilon
            
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        
        # Q-value 계산
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.model(state_tensor.unsqueeze(0)).cpu().numpy()[0]
            
        return np.argmax(q_values)  # 최대 Q-value를 가진 액션 선택

    def remember(self, state, action, reward, next_state, done):
        """
        경험을 메모리에 저장
        
        Args:
            state (numpy.ndarray): 현재 상태
            action (int): 수행한 행동
            reward (float): 받은 보상
            next_state (numpy.ndarray): 다음 상태
            done (bool): 에피소드 종료 여부
        """
        # 메모리 크기 제한
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # 오래된 메모리 제거
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        경험 리플레이를 통한 학습
        
        Args:
            batch_size (int): 배치 크기
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        # 각 요소를 개별적으로 처리
        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch])

        # 텐서로 변환
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Double DQN 구현 - 현재 네트워크로 행동 선택, 타겟 네트워크로 가치 평가
        indices = torch.argmax(self.model(next_states), dim=1)
        next_q_values = self.target_model(next_states).gather(1, indices.unsqueeze(1)).squeeze(1)
        
        # Q 타겟 계산
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 현재 Q 값 계산
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 손실 계산 및 업데이트
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 타겟 네트워크 주기적 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()
        
        # 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        모델 가중치 로드
        
        Args:
            name (str): 모델 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(name):
                logging.warning(f"모델 파일이 존재하지 않습니다: {name}")
                return False
                
            self.model.load_state_dict(torch.load(name, map_location=device))
            self.update_target_model()  # 타겟 모델도 업데이트
            logging.info(f"모델 로드 성공: {name}")
            return True
        except Exception as e:
            logging.error(f"모델 로드 오류: {e}")
            return False

    def save(self, name):
        """
        모델 가중치 저장
        
        Args:
            name (str): 저장할 파일 경로
        """
        os.makedirs(os.path.dirname(name), exist_ok=True)
        torch.save(self.model.state_dict(), name)
        logging.info(f"모델 저장됨: {name}")