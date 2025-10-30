import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

class StockTradingEnvironment:
    """
    A reinforcement learning environment for stock trading.
    
    Attributes:
        data (pd.DataFrame): Stock data with OHLC (Open, High, Low, Close) and Volume columns.
        initial_balance (float): Initial cash balance for trading.
        balance (float): Current cash balance.
        shares_held (float): Number of shares currently held.
        current_step (int): Current time step in the data.
        max_steps (int): Maximum number of steps based on data length.
        done (bool): Indicates whether the environment is done.
        transaction_cost (float): 거래 비용 (매수/매도 시 적용되는 수수료)
        holding_incentive (float): 홀딩 인센티브 (주식 보유 시 추가 보상)
        holding_period (int): 현재 주식 보유 기간
        last_action (int): 이전 행동
    """

    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0,
                 transaction_cost: float = 0.0005, holding_incentive: float = 0.0):
        """
        Initialize the stock trading environment.

        Args:
            data (pd.DataFrame): DataFrame containing stock OHLC data.
            initial_balance (float, optional): Starting cash balance. Defaults to 100000.0.
            transaction_cost (float, optional): 거래 비용 (매수/매도 시 적용). Defaults to 0.0005 (0.05%).
            holding_incentive (float, optional): 홀딩 인센티브 (주식 보유 시 추가 보상). Defaults to 0.0.
        """
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close', and 'Volume' columns")
        
        self.data = data.reset_index(drop=True)  # Ensure index is numeric for iloc
        self.initial_balance = float(initial_balance)
        self.balance = self.initial_balance
        self.shares_held = 0.0  # Allow fractional shares
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.done = False  # Initialize done attribute
        
        # 투자 스타일에 따른 설정
        self.transaction_cost = transaction_cost
        self.holding_incentive = holding_incentive
        self.holding_period = 0
        self.last_action = 0  # 초기값: Hold

        logging.info(f"StockTradingEnvironment initialized with transaction cost: {transaction_cost:.4f}, holding incentive: {holding_incentive:.4f}")

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.

        Returns:
            np.ndarray: Initial state array.
        """
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = 0
        self.done = False  # Reset done attribute
        self.holding_period = 0
        self.last_action = 0
        logging.debug("Environment reset")
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: State array containing [close_price, balance, shares_held].
        """
        try:
            # Access closing price as a scalar using .item() to avoid FutureWarning
            current_price = self.data['Close'].iloc[self.current_step].item()
            balance = float(self.balance)  # Ensure scalar float
            shares_held = float(self.shares_held)  # Ensure scalar float
            # Create state array with consistent scalar values
            state = np.array([
                current_price / self.data['Close'].iloc[0].item() - 1,  # 가격 변화율
                self.balance / self.initial_balance - 1,  # 잔고 변화율
                self.shares_held * current_price / self.initial_balance  # 보유 주식 가치 비율
            ], dtype=np.float32)
            return state
        except Exception as e:
            logging.error(f"Error in _get_observation: {e}")
            print(f"Current step: {self.current_step}")
            print(f"Type of current_price: {type(current_price)}, Value: {current_price}")
            print(f"Type of balance: {type(self.balance)}, Value: {self.balance}")
            print(f"Type of shares_held: {type(self.shares_held)}, Value: {self.shares_held}")
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): Action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            Tuple[np.ndarray, float, bool]: (next_state, reward, done).
        """
        self.done = self.current_step >= self.max_steps
        
        if self.done:
            return self._get_observation(), 0, self.done

        # 현재 주가
        current_price = self.data['Close'].iloc[self.current_step].item()
        
        # 이전 포트폴리오 가치 (액션 전)
        prev_portfolio_value = self.balance + self.shares_held * current_price
        
        # 액션 실행
        transaction_fee = 0.0
        
        if action == 1:  # Buy
            if self.balance >= current_price:
                # 가능한 모든 주식 구매 (최대 80%까지만 사용)
                max_purchase = self.balance * 0.8
                shares_to_buy = (max_purchase // current_price) if current_price > 0 else 0
                
                if shares_to_buy > 0:
                    purchase_cost = shares_to_buy * current_price
                    transaction_fee = purchase_cost * self.transaction_cost
                    
                    self.shares_held += shares_to_buy
                    self.balance -= (purchase_cost + transaction_fee)
                    self.holding_period = 0  # 새로 구매했으므로 보유 기간 리셋
                    
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # 모든 주식 판매
                sell_value = self.shares_held * current_price
                transaction_fee = sell_value * self.transaction_cost
                
                self.balance += (sell_value - transaction_fee)
                self.shares_held = 0.0
                self.holding_period = 0  # 판매했으므로 보유 기간 리셋
        else:  # Hold
            if self.shares_held > 0:
                self.holding_period += 1  # 보유 기간 증가
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            self.done = True
            return self._get_observation(), 0, self.done
            
        next_price = self.data['Close'].iloc[self.current_step].item()
        
        # 현재 포트폴리오 가치 (액션 후)
        current_portfolio_value = self.balance + self.shares_held * next_price
        
        # 기본 보상: 포트폴리오 가치 변화율
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # 거래 비용 페널티
        reward -= transaction_fee / self.initial_balance
        
        # 홀딩 인센티브 (주식을 보유한 경우에만)
        if self.shares_held > 0 and action == 0:  # Hold 액션
            # 보유 기간에 따라 스케일링된 홀딩 인센티브
            scaled_holding_incentive = self.holding_incentive * min(self.holding_period / 10, 1.0)
            reward += scaled_holding_incentive
        
        self.last_action = action
        
        return self._get_observation(), reward, self.done

    def render(self, mode: str = 'human') -> None:
        """
        Render the current state of the environment.

        Args:
            mode (str, optional): Rendering mode. Defaults to 'human'.
        """
        current_price = self.data['Close'].iloc[self.current_step].item()
        portfolio_value = self.balance + self.shares_held * current_price
        action_name = ["보유", "매수", "매도"][self.last_action]
        
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, "
              f"Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}, "
              f"Portfolio Value: {portfolio_value:.2f}, Action: {action_name}, "
              f"Holding Period: {self.holding_period}")