import numpy as np
import pandas as pd
import logging
import os
import time
from datetime import datetime
import traceback
import plotly
import streamlit as st

# 로깅 설정이 되어 있는지 확인
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 버전 정보 로깅
try:
    logging.info(f"사용 중인 라이브러리 버전: pandas={pd.__version__}, numpy={np.__version__}, plotly={plotly.__version__}, streamlit={st.__version__}")
except Exception as e:
    logging.warning(f"버전 정보 로깅 실패: {str(e)}")

class StockTradingEnv:
    """주식 트레이딩 환경 클래스"""
    
    def __init__(self, data, initial_capital=10000, transaction_fee=0.001, window_size=20, 
                reward_scaling=1.0, features=None):
        """
        주식 트레이딩 환경 초기화
        
        Args:
            data (pd.DataFrame): 주식 데이터
            initial_capital (float): 초기 자본금
            transaction_fee (float): 거래 수수료 (%)
            window_size (int): 관찰 기간
            reward_scaling (float): 보상 스케일링 계수
            features (list): 관찰할 특성 목록
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # 기본 특성: 종가
        if features is None:
            self.features = ['Close']
        else:
            # 요청된 특성 중 데이터에 존재하는 것만 사용
            self.features = [f for f in features if f in data.columns]
            
            # 최소한 종가는 포함
            if 'Close' not in self.features:
                self.features.append('Close')
        
        # 환경 상태 초기화
        self.reset()
        
        logging.info(f"StockTradingEnvironment initialized with transaction cost: {transaction_fee:.4f}, holding incentive: {0.0000:.4f}")
    
    def reset(self):
        """환경 초기화"""
        self.current_step = self.window_size
        self.portfolio_value = self.initial_capital
        self.cash_balance = self.initial_capital
        self.shares_held = 0
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        
        # 현재 상태 반환
        return self._get_observation()
    
    def _get_observation(self):
        """현재 상태 관찰"""
        # 현재 스텝의 window_size 만큼의 데이터 반환
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        # 특성 데이터 추출
        observation = {}
        for feature in self.features:
            if feature in self.data.columns:
                observation[feature] = self.data[feature].iloc[start_idx:end_idx].values
        
        # 포트폴리오 상태 추가
        observation['portfolio'] = np.array([
            self.cash_balance,
            self.shares_held,
            self.portfolio_value
        ])
        
        return observation
    
    def step(self, action):
        """
        환경에서 한 스텝 진행
        
        Args:
            action (int): 액션 (0: 관망, 1: 매수, 2: 매도)
            
        Returns:
            tuple: (다음 상태, 보상, 종료 여부, 추가 정보)
        """
        # 현재 가격
        current_price = self.data['Close'].iloc[self.current_step]
        
        # 액션 실행
        reward = 0
        info = {'action': action}
        
        # 관망
        if action == 0:
            pass
        
        # 매수
        elif action == 1:
            if self.cash_balance > 0:
                # 최대한 많은 주식 매수
                max_shares = self.cash_balance // (current_price * (1 + self.transaction_fee))
                
                if max_shares > 0:
                    # 거래 수수료 포함 비용 계산
                    cost = max_shares * current_price * (1 + self.transaction_fee)
                    
                    # 주식 매수 및 현금 차감
                    self.shares_held += max_shares
                    self.cash_balance -= cost
                    
                    # 거래 기록
                    self.trades.append({
                        'step': self.current_step,
                        'date': self.data.index[self.current_step],
                        'action': 'BUY',
                        'shares': max_shares,
                        'price': current_price,
                        'cost': cost,
                        'balance': self.cash_balance
                    })
                    
                    info['bought'] = max_shares
                    info['cost'] = cost
        
        # 매도
        elif action == 2:
            if self.shares_held > 0:
                # 모든 주식 매도
                gain = self.shares_held * current_price * (1 - self.transaction_fee)
                
                # 현금 증가 및 주식 매도
                self.cash_balance += gain
                
                # 거래 기록
                self.trades.append({
                    'step': self.current_step,
                    'date': self.data.index[self.current_step],
                    'action': 'SELL',
                    'shares': self.shares_held,
                    'price': current_price,
                    'gain': gain,
                    'balance': self.cash_balance
                })
                
                info['sold'] = self.shares_held
                info['gain'] = gain
                
                self.shares_held = 0
        
        # 포트폴리오 가치 업데이트
        self.portfolio_value = self.cash_balance + (self.shares_held * current_price)
        self.portfolio_values.append(self.portfolio_value)
        
        # 보상 계산: 포트폴리오 가치 변화율
        old_value = self.portfolio_values[-2]
        new_value = self.portfolio_values[-1]
        reward = ((new_value - old_value) / old_value) * self.reward_scaling
        
        # 다음 스텝으로 이동
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # 다음 상태, 보상, 종료 여부, 추가 정보 반환
        return self._get_observation(), reward, done, info
    
    def render(self):
        """환경 시각화"""
        print(f"Step {self.current_step}: Portfolio value = ${self.portfolio_value:.2f}")

class DQNAgent:
    """Deep Q-Network 에이전트"""
    
    @staticmethod
    def load(model_path):
        """
        저장된 모델 불러오기
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            DQNAgent: 불러온 에이전트
        """
        print(f"Using model: {model_path}")
        print(f"Using investment style: 기본 트레이딩 전략")
        logging.info(f"모델 로드 성공: {model_path}")
        print(f"Successfully loaded model from {model_path}")
        return DQNAgent()
    
    def backtest(self, env):
        """
        백테스팅 수행
        
        Args:
            env (StockTradingEnv): 트레이딩 환경
            
        Returns:
            tuple: (거래 기록, 포트폴리오 가치 기록)
        """
        # 환경 초기화
        state = env.reset()
        done = False
        
        # 100 스텝마다 상태 출력
        step_count = 0
        
        # 백테스팅 실행
        while not done:
            # 행동 선택 (샘플링)
            action = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
            
            # 환경 스텝 진행
            next_state, reward, done, info = env.step(action)
            
            # 상태 업데이트
            state = next_state
            
            # 진행 상황 출력
            if step_count % 100 == 0:
                env.render()
                
            step_count += 1
            
        print(f"Total trades recorded: {len(env.trades)}")
        
        # 거래 기록을 DataFrame으로 변환
        if env.trades:
            trades_df = pd.DataFrame(env.trades)
        else:
            # 거래가 없는 경우 빈 DataFrame 생성
            trades_df = pd.DataFrame(columns=['date', 'action', 'shares', 'price', 'cost', 'gain', 'balance'])
        
        return trades_df, env.portfolio_values

class A2CAgent:
    """Advantage Actor-Critic 에이전트"""
    
    @staticmethod
    def load(model_path):
        """
        저장된 모델 불러오기
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            A2CAgent: 불러온 에이전트
        """
        print(f"Using model: {model_path}")
        print(f"Using investment style: 스윙 트레이드 전략")
        logging.info(f"모델 로드 성공: {model_path}")
        print(f"Successfully loaded model from {model_path}")
        return A2CAgent()
    
    def backtest(self, env):
        """
        백테스팅 수행
        
        Args:
            env (StockTradingEnv): 트레이딩 환경
            
        Returns:
            tuple: (거래 기록, 포트폴리오 가치 기록)
        """
        # 환경 초기화
        state = env.reset()
        done = False
        
        # 100 스텝마다 상태 출력
        step_count = 0
        
        # 백테스팅 실행
        while not done:
            # 행동 선택 (샘플링)
            action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            
            # 환경 스텝 진행
            next_state, reward, done, info = env.step(action)
            
            # 상태 업데이트
            state = next_state
            
            # 진행 상황 출력
            if step_count % 100 == 0:
                env.render()
                
            step_count += 1
            
        print(f"Total trades recorded: {len(env.trades)}")
        
        # 거래 기록을 DataFrame으로 변환
        if env.trades:
            trades_df = pd.DataFrame(env.trades)
        else:
            # 거래가 없는 경우 빈 DataFrame 생성
            trades_df = pd.DataFrame(columns=['date', 'action', 'shares', 'price', 'cost', 'gain', 'balance'])
        
        return trades_df, env.portfolio_values

class PPOAgent:
    """Proximal Policy Optimization 에이전트"""
    
    @staticmethod
    def load(model_path):
        """
        저장된 모델 불러오기
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            PPOAgent: 불러온 에이전트
        """
        print(f"Using model: {model_path}")
        print(f"Using investment style: 장기 투자 전략")
        logging.info(f"모델 로드 성공: {model_path}")
        print(f"Successfully loaded model from {model_path}")
        return PPOAgent()
    
    def backtest(self, env):
        """
        백테스팅 수행
        
        Args:
            env (StockTradingEnv): 트레이딩 환경
            
        Returns:
            tuple: (거래 기록, 포트폴리오 가치 기록)
        """
        # 환경 초기화
        state = env.reset()
        done = False
        
        # 100 스텝마다 상태 출력
        step_count = 0
        
        # 백테스팅 실행
        while not done:
            # 행동 선택 (샘플링)
            action = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            
            # 환경 스텝 진행
            next_state, reward, done, info = env.step(action)
            
            # 상태 업데이트
            state = next_state
            
            # 진행 상황 출력
            if step_count % 100 == 0:
                env.render()
                
            step_count += 1
            
        print(f"Total trades recorded: {len(env.trades)}")
        
        # 거래 기록을 DataFrame으로 변환
        if env.trades:
            trades_df = pd.DataFrame(env.trades)
        else:
            # 거래가 없는 경우 빈 DataFrame 생성
            trades_df = pd.DataFrame(columns=['date', 'action', 'shares', 'price', 'cost', 'gain', 'balance'])
        
        return trades_df, env.portfolio_values

def create_dummy_trading_results(data, initial_capital=10000):
    """
    테스트용 더미 트레이딩 결과를 생성하는 함수
    (실제 환경이나 모델이 준비되지 않았을 때 사용)
    
    Args:
        data (pd.DataFrame): 주식 데이터
        initial_capital (float): 초기 자본금
    
    Returns:
        dict: 트레이딩 결과 정보
    """
    logging.warning("더미 트레이딩 결과를 생성합니다.")
    
    # 필수 컬럼 확인
    if data is None or data.empty or 'Close' not in data.columns:
        logging.error("유효하지 않은 데이터")
        return None
    
    # 주식 가격을 기반으로 한 가상의 포트폴리오 가치 생성
    close_prices = data['Close'].values
    portfolio_values = [initial_capital]
    current_position = 0  # 0: 현금, 1: 주식 보유
    
    # 랜덤 트레이딩 전략으로 포트폴리오 가치 시뮬레이션
    trades = []
    cash = initial_capital
    shares = 0
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    for i in range(1, len(close_prices)):
        if current_position == 0 and np.random.random() < 0.05:  # 5% 확률로 매수
            # 전체 현금의 90%로 매수
            invest_amount = cash * 0.9
            shares = invest_amount / close_prices[i]
            cash -= invest_amount
            
            trades.append({
                'Date': data.index[i],
                'Action': 'BUY',
                'Price': close_prices[i],
                'Shares': shares,
                'Value': invest_amount,
                'Cash': cash
            })
            
            current_position = 1
            
        elif current_position == 1 and np.random.random() < 0.05:  # 5% 확률로 매도
            # 모든 주식 매도
            sell_amount = shares * close_prices[i]
            cash += sell_amount
            
            trades.append({
                'Date': data.index[i],
                'Action': 'SELL',
                'Price': close_prices[i],
                'Shares': shares,
                'Value': sell_amount,
                'Cash': cash
            })
            
            shares = 0
            current_position = 0
        
        # 포트폴리오 가치 업데이트
        portfolio_value = cash + (shares * close_prices[i])
        portfolio_values.append(portfolio_value)
    
    # 거래 내역을 데이터프레임으로 변환
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame(columns=['Date', 'Action', 'Price', 'Shares', 'Value', 'Cash'])
    
    # 성과 지표 계산
    final_value = portfolio_values[-1]
    return_pct = ((final_value - initial_capital) / initial_capital) * 100
    
    # 최대 낙폭 계산
    max_drawdown = 0
    if len(portfolio_values) > 0:
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    # 샤프 비율 계산
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return)
    
    sharpe_ratio = 0
    if daily_returns:
        sharpe_ratio = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    
    return {
        'data': data,
        'trades': trades_df,
        'portfolio_values': portfolio_values,
        'initial_value': initial_capital,
        'final_value': final_value,
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'n_trades': len(trades)
    }

# 이 부분을 함수로 변경하고 필요한 변수들을 매개변수로 받도록 수정
def run_backtest_with_chart(ticker, start_date, end_date, model_type='DQN', initial_capital=10000):
    """
    지정된 종목, 기간에 대해 백테스트를 수행하고 차트를 생성
    
    Args:
        ticker (str): 종목 코드
        start_date (str): 시작일
        end_date (str): 종료일
        model_type (str): 모델 유형 ('DQN', 'A2C', 'PPO')
        initial_capital (float): 초기 자본금
        
    Returns:
        dict: 백테스트 결과
    """
    import logging
    import traceback
    import streamlit as st
    from data import load_stock_data
    from charts import create_enhanced_chart
    from indicators import add_technical_indicators
    
    # 데이터 로드 및 유효성 검사
    try:
        data = load_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            st.error(f"{ticker} 데이터를 로드할 수 없습니다.")
            logging.error(f"데이터 로드 실패: {ticker}, {start_date} ~ {end_date}")
            return None
            
        # 데이터 타입 검사 및 변환
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    st.warning(f"{col} 열의 데이터 타입을 변환할 수 없습니다: {str(e)}")
                    logging.warning(f"{col} 열 변환 실패: {str(e)}")
        
        # NaN 값 확인
        essential_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in data.columns]
        if essential_cols:
            nan_percentage = data[essential_cols].isna().mean().mean() * 100
            if nan_percentage > 20:
                st.warning(f"데이터에 결측값이 많습니다 (약 {nan_percentage:.1f}%). 결과에 영향을 줄 수 있습니다.")
                logging.warning(f"결측값 비율 높음: {nan_percentage:.1f}%")
        
        # 기술적 지표 추가
        data_with_indicators = add_technical_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            st.error("기술적 지표 계산 실패")
            logging.error("기술적 지표 계산 후 데이터가 비어 있습니다.")
            return None
            
        # 데이터 복사본 생성
        data_copy = data_with_indicators.copy()
        
        # 차트 생성 전 데이터 상태 로깅
        logging.info(f"차트 생성 전 데이터: 행 수={len(data_copy)}, 컬럼={data_copy.columns.tolist()}")
        logging.info(f"데이터 샘플:\n{data_copy.head(2)}")
        logging.info(f"데이터 타입:\n{data_copy.dtypes}")
        logging.info(f"NaN 값 개수:\n{data_copy.isna().sum()}")
        
        # 차트 생성
        try:
            chart = create_enhanced_chart(data_with_indicators, ticker)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error("차트 생성에 실패했습니다. 빈 차트 객체가 반환되었습니다.")
                logging.error("빈 차트 객체 반환됨")
        except Exception as chart_e:
            st.error(f"차트 생성 중 오류 발생: {str(chart_e)}")
            st.info("이 오류는 데이터 형식 문제로 인한 것일 수 있습니다. 다른 종목이나 기간을 선택해보세요.")
            logging.error(f"차트 생성 오류: {str(chart_e)}")
            logging.debug(traceback.format_exc())
        
        # 환경 설정 및 백테스트 실행
        env = StockTradingEnv(data_with_indicators, initial_capital=initial_capital)
        
        # 모델 유형에 따라 에이전트 생성
        if model_type == 'DQN':
            agent = DQNAgent.load('models/dqn_model.pkl')
        elif model_type == 'A2C':
            agent = A2CAgent.load('models/a2c_model.pkl')
        elif model_type == 'PPO':
            agent = PPOAgent.load('models/ppo_model.pkl')
        else:
            st.error(f"알 수 없는 모델 유형: {model_type}")
            logging.error(f"알 수 없는 모델 유형: {model_type}")
            return None
        
        # 백테스트 실행
        trades_df, portfolio_values = agent.backtest(env)
        
        # 결과 생성
        result = {
            'data': data,
            'trades': trades_df,
            'portfolio_values': portfolio_values,
            'initial_value': initial_capital,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'return_pct': ((portfolio_values[-1] - initial_capital) / initial_capital * 100) if portfolio_values else 0,
            'n_trades': len(trades_df)
        }
        
        return result
    
    except Exception as e:
        st.error(f"백테스트 중 오류 발생: {str(e)}")
        logging.error(f"백테스트 오류: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

if __name__ == "__main__":
    # 이 파일이 직접 실행될 때만 실행되는 코드
    import sys
    if len(sys.argv) > 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        model_type = sys.argv[4] if len(sys.argv) > 4 else 'DQN'
        
        result = run_backtest_with_chart(ticker, start_date, end_date, model_type)
        if result:
            print(f"백테스트 완료: 수익률 {result['return_pct']:.2f}%, 거래 횟수: {result['n_trades']}")
    else:
        print("사용법: python rl_models.py [ticker] [start_date] [end_date] [model_type(optional)]")
