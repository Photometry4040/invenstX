"""
Enhanced Stock Trading Environment with Technical Indicators

확장된 State representation을 사용하는 강화학습 환경
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional


class EnhancedStockTradingEnvironment:
    """
    기술적 지표를 포함한 확장된 주식 거래 환경

    State: 23차원
    - Close, Balance, Shares (기본 3개)
    - RSI, MACD, MACD_Signal, MACD_Histogram (모멘텀 4개)
    - BB_Upper, BB_Middle, BB_Lower (변동성 3개)
    - MA_5, MA_20, MA_50 (트렌드 3개)
    - ATR (변동성 1개)
    - Volume_MA, Volume_Ratio (거래량 2개)
    - Daily_Return, Log_Return (수익률 2개)
    - Momentum_5, Momentum_20 (모멘텀 2개)
    - Volatility_20 (변동성 1개)
    """

    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.0005,
                 holding_incentive: float = 0.0,
                 use_technical_indicators: bool = True):
        """
        환경 초기화

        Args:
            data: 피처가 추가된 데이터프레임
            initial_balance: 초기 자본
            transaction_cost: 거래 비용
            holding_incentive: 보유 인센티브
            use_technical_indicators: 기술적 지표 사용 여부
        """
        # 데이터 검증
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain {required_columns}")

        self.data = data.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.transaction_cost = transaction_cost
        self.holding_incentive = holding_incentive
        self.use_technical_indicators = use_technical_indicators

        # 상태 변수
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.done = False
        self.holding_period = 0
        self.last_action = 0

        # 기술적 지표 컬럼 확인
        self.indicator_columns = []
        if use_technical_indicators:
            potential_indicators = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'MA_5', 'MA_20', 'MA_50',
                'ATR', 'Volume_MA', 'Volume_Ratio',
                'Daily_Return', 'Log_Return',
                'Momentum_5', 'Momentum_20', 'Volatility_20'
            ]
            self.indicator_columns = [col for col in potential_indicators if col in data.columns]

        state_dim = 3 + len(self.indicator_columns)
        logging.info(f"EnhancedStockTradingEnvironment initialized")
        logging.info(f"  State dimension: {state_dim} (3 basic + {len(self.indicator_columns)} indicators)")
        logging.info(f"  Transaction cost: {transaction_cost:.4f}")
        logging.info(f"  Holding incentive: {holding_incentive:.4f}")
        logging.info(f"  Data length: {len(data)}")

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = 0
        self.done = False
        self.holding_period = 0
        self.last_action = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        현재 상태 반환

        Returns:
            State 배열 (3 + 기술적 지표 개수)
        """
        try:
            row = self.data.iloc[self.current_step]

            # 기본 상태 (3차원)
            current_price = float(row['Close'])
            initial_price = float(self.data['Close'].iloc[0])

            basic_state = [
                current_price / initial_price - 1,  # 가격 변화율
                self.balance / self.initial_balance - 1,  # 잔고 변화율
                self.shares_held * current_price / self.initial_balance  # 보유 비율
            ]

            # 기술적 지표 추가
            if self.use_technical_indicators and self.indicator_columns:
                # 정규화된 지표 값
                indicators = []
                for col in self.indicator_columns:
                    value = float(row[col])

                    # 지표별 정규화
                    if col == 'RSI':
                        # RSI는 0-100 범위 → -1 to 1
                        normalized = (value / 50) - 1
                    elif col in ['MACD', 'MACD_Signal', 'MACD_Histogram']:
                        # MACD는 가격 대비 비율
                        normalized = value / current_price if current_price != 0 else 0
                    elif col in ['BB_Upper', 'BB_Middle', 'BB_Lower', 'MA_5', 'MA_20', 'MA_50']:
                        # 이동평균은 현재 가격 대비 비율
                        normalized = (value / current_price - 1) if current_price != 0 else 0
                    elif col == 'ATR':
                        # ATR은 가격 대비 비율
                        normalized = value / current_price if current_price != 0 else 0
                    elif col == 'Volume_Ratio':
                        # 거래량 비율은 그대로 (단, 클리핑)
                        normalized = np.clip(value - 1, -5, 5) / 5
                    elif col == 'Volume_MA':
                        # 거래량 MA는 현재 거래량 대비
                        current_volume = float(row['Volume'])
                        normalized = (current_volume / value - 1) if value != 0 else 0
                    elif col in ['Daily_Return', 'Log_Return']:
                        # 수익률은 그대로 (단, 클리핑)
                        normalized = np.clip(value, -0.2, 0.2) / 0.2
                    elif col in ['Momentum_5', 'Momentum_20']:
                        # 모멘텀은 가격 대비
                        normalized = value / current_price if current_price != 0 else 0
                    elif col == 'Volatility_20':
                        # 변동성은 그대로 (단, 클리핑)
                        normalized = np.clip(value, 0, 0.1) / 0.1
                    else:
                        normalized = value

                    # NaN 체크 및 대체
                    if np.isnan(normalized) or np.isinf(normalized):
                        normalized = 0.0

                    indicators.append(normalized)

                state = basic_state + indicators
            else:
                state = basic_state

            return np.array(state, dtype=np.float32)

        except Exception as e:
            logging.error(f"Error in _get_observation at step {self.current_step}: {e}")
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        행동 수행

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            (next_state, reward, done)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        current_price = float(self.data['Close'].iloc[self.current_step])

        # 행동 수행
        reward = 0
        if action == 1:  # Buy
            if self.balance > 0:
                # 거래 비용 고려
                shares_to_buy = self.balance / (current_price * (1 + self.transaction_cost))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)

                if cost <= self.balance:
                    self.shares_held += shares_to_buy
                    self.balance -= cost
                    self.holding_period = 0
                    logging.debug(f"Buy: {shares_to_buy:.2f} shares at ${current_price:.2f}")

        elif action == 2:  # Sell
            if self.shares_held > 0:
                # 거래 비용 고려
                proceeds = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                reward = proceeds - (self.shares_held * current_price)  # 거래 비용만큼 패널티
                self.shares_held = 0
                self.holding_period = 0
                logging.debug(f"Sell: all shares at ${current_price:.2f}")

        # 보유 인센티브
        if self.shares_held > 0:
            self.holding_period += 1
            reward += self.holding_incentive * self.shares_held * current_price

        # 다음 스텝
        self.current_step += 1
        self.last_action = action

        # 종료 조건
        if self.current_step >= self.max_steps:
            self.done = True
            # 마지막에 주식 보유 중이면 강제 매도
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0

        # 포트폴리오 가치 기반 보상
        portfolio_value = self.balance + self.shares_held * current_price
        reward += (portfolio_value - self.initial_balance) / self.initial_balance * 0.01

        next_state = self._get_observation()

        return next_state, reward, self.done

    def get_portfolio_value(self) -> float:
        """현재 포트폴리오 가치"""
        current_price = float(self.data['Close'].iloc[self.current_step])
        return self.balance + self.shares_held * current_price

    def get_state_dimension(self) -> int:
        """State 차원"""
        return 3 + len(self.indicator_columns)


# 호환성을 위한 래퍼
class StockTradingEnvironment(EnhancedStockTradingEnvironment):
    """기존 코드와의 호환성을 위한 래퍼 클래스"""
    pass
