"""
통합 트레이딩 시스템
Paper Trading을 위한 모든 모듈 통합
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json

# 구현된 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent / 'implementations'))

from implementations.paper_trading_engine import (
    PaperTradingEngine, Order, OrderType, OrderSide, OrderStatus
)
from implementations.advanced_risk_management import AdvancedRiskManager
from reinforcement_learning.model import DQNAgent
from reinforcement_learning.environment import StockTradingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """
    통합 트레이딩 시스템
    - Paper Trading 엔진
    - 리스크 관리
    - RL 모델 통합
    - 성과 모니터링
    """

    def __init__(self,
                 initial_balance: float = 100000,
                 symbols: List[str] = None,
                 models_dir: str = 'models'):
        """
        Args:
            initial_balance: 초기 자본
            symbols: 거래할 종목 리스트
            models_dir: 모델 저장 디렉토리
        """
        self.initial_balance = initial_balance
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT']
        self.models_dir = Path(models_dir)

        # Paper Trading 엔진
        self.paper_trading = PaperTradingEngine(
            initial_balance=initial_balance,
            commission_rate=0.001,
            min_commission=1.0
        )

        # 리스크 관리자
        self.risk_manager = AdvancedRiskManager()

        # RL 모델 (심볼별)
        self.agents: Dict[str, DQNAgent] = {}

        # 시장 데이터 캐시
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}

        # 성과 추적
        self.trade_log = []
        self.daily_pnl = []
        self.performance_metrics = {}

        # 실행 상태
        self.running = False
        self.start_time = None

        logger.info(f"Integrated Trading System initialized with ${initial_balance:,.2f}")
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")

    def load_models(self):
        """학습된 RL 모델 로드"""
        for symbol in self.symbols:
            model_path = self.models_dir / f"{symbol}_default_enhanced_model.pth"

            if not model_path.exists():
                logger.warning(f"Model not found for {symbol}: {model_path}")
                # 기본 모델 생성
                self.agents[symbol] = DQNAgent(
                    state_size=23,
                    action_size=3,
                    epsilon=0.0  # Paper Trading에서는 탐험 없음
                )
            else:
                import torch
                agent = DQNAgent(state_size=23, action_size=3, epsilon=0.0)
                agent.model.load_state_dict(torch.load(model_path))
                agent.model.eval()
                self.agents[symbol] = agent
                logger.info(f"Loaded model for {symbol}")

    def load_market_data(self, start_date: str, end_date: str):
        """시장 데이터 로드"""
        logger.info(f"Loading market data from {start_date} to {end_date}")

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    logger.error(f"No data for {symbol}")
                    continue

                self.market_data[symbol] = df
                self.current_prices[symbol] = df['Close'].iloc[-1]

                logger.info(f"Loaded {len(df)} days of data for {symbol}")

            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

    def calculate_features(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """기술적 지표 계산"""
        features = []

        # 기본 가격 정보
        close = df['Close'].values
        volume = df['Volume'].values

        # 최근 데이터만 사용 (메모리 절약)
        if len(close) > 100:
            close = close[-100:]
            volume = volume[-100:]

        # 가격 정규화
        if len(close) > 0:
            price_norm = close[-1] / close[0] if close[0] > 0 else 1

        # 기본 특징 (3개)
        balance = self.paper_trading.cash
        positions = self.paper_trading.get_positions()
        shares_held = positions.get(symbol, {}).get('quantity', 0) if isinstance(positions.get(symbol), dict) else 0

        features.extend([
            price_norm if len(close) > 0 else 1,
            balance / self.initial_balance,
            shares_held / 100  # 정규화
        ])

        # 기술적 지표 (20개)
        if len(close) >= 20:
            # SMA
            sma_5 = np.mean(close[-5:])
            sma_10 = np.mean(close[-10:])
            sma_20 = np.mean(close[-20:])

            # RSI
            gains = []
            losses = []
            for i in range(1, min(15, len(close))):
                change = close[-i] - close[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 50

            # 볼린저 밴드
            std_20 = np.std(close[-20:])
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

            # 변동성
            returns = np.diff(close[-20:]) / close[-21:-1]
            volatility = np.std(returns)

            # MACD (간단 버전)
            ema_12 = np.mean(close[-12:])
            ema_26 = np.mean(close[-26:]) if len(close) >= 26 else np.mean(close)
            macd = ema_12 - ema_26
            signal = np.mean([macd])  # 간단히 구현

            # 모멘텀
            momentum_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
            momentum_10 = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0

            # 거래량
            volume_sma = np.mean(volume[-20:])
            volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1

            features.extend([
                close[-1] / sma_5 - 1 if sma_5 > 0 else 0,
                close[-1] / sma_10 - 1 if sma_10 > 0 else 0,
                close[-1] / sma_20 - 1 if sma_20 > 0 else 0,
                rsi / 100,
                bb_position,
                volatility,
                macd / close[-1] if close[-1] > 0 else 0,
                signal / close[-1] if close[-1] > 0 else 0,
                momentum_5,
                momentum_10,
                np.log1p(volume_ratio),
                # 추가 특징들 (총 20개까지)
                (close[-1] - np.min(close[-20:])) / (np.max(close[-20:]) - np.min(close[-20:])) if (np.max(close[-20:]) - np.min(close[-20:])) > 0 else 0.5,
                np.mean(returns),
                np.max(returns),
                np.min(returns),
                (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0,
                std_20 / close[-1] if close[-1] > 0 else 0,
                (bb_upper - bb_lower) / close[-1] if close[-1] > 0 else 0,
                (close[-1] - close[-5]) / close[-5] if len(close) >= 5 and close[-5] > 0 else 0,
                (volume[-1] - volume[-5]) / volume[-5] if len(volume) >= 5 and volume[-5] > 0 else 0,
            ])
        else:
            # 데이터 부족 시 기본값
            features.extend([0] * 20)

        return np.array(features[:23])  # 정확히 23개

    async def run_paper_trading(self, days: int = 7):
        """Paper Trading 실행"""
        logger.info(f"Starting {days}-day Paper Trading simulation")

        self.running = True
        self.start_time = datetime.now()

        # 모델 로드
        self.load_models()

        # 과거 데이터 로드 (백테스팅 방식)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)  # 기술적 지표 계산을 위한 추가 데이터

        self.load_market_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # 일별 시뮬레이션
        for day_idx in range(days):
            trading_date = end_date - timedelta(days=days - day_idx - 1)
            logger.info(f"\n{'='*60}")
            logger.info(f"Trading Day {day_idx + 1}/{days} - {trading_date.strftime('%Y-%m-%d')}")
            logger.info(f"{'='*60}")

            # 각 종목에 대해 거래 결정
            for symbol in self.symbols:
                await self.process_symbol(symbol, day_idx)

            # 포트폴리오 가치 업데이트
            await self.paper_trading.update_market_prices(self.current_prices)

            # 일일 성과 기록
            self.record_daily_performance()

            # 상태 출력
            self.print_status()

            # 짧은 대기 (시뮬레이션)
            await asyncio.sleep(0.1)

        # 최종 결과
        self.generate_final_report()

    async def process_symbol(self, symbol: str, day_idx: int):
        """종목별 거래 처리"""
        if symbol not in self.market_data or symbol not in self.agents:
            return

        df = self.market_data[symbol]
        if len(df) <= day_idx + 100:
            return

        # 해당 날짜까지의 데이터
        current_df = df.iloc[:-(len(df) - day_idx - 100)] if day_idx < len(df) - 100 else df

        if len(current_df) < 20:
            return

        # 특징 계산
        features = self.calculate_features(symbol, current_df)

        # RL 모델 예측
        agent = self.agents[symbol]
        action = agent.act(features.reshape(1, -1))

        # 현재 가격
        current_price = current_df['Close'].iloc[-1]
        self.current_prices[symbol] = current_price

        # 리스크 분석
        risk_metrics = self.risk_manager.analyze_trade(
            symbol=symbol,
            entry_price=current_price,
            confidence=0.8,  # 기본 신뢰도
            market_data=current_df,
            portfolio_value=self.paper_trading.get_portfolio_value(),
            current_positions=self.paper_trading.get_positions()
        )

        # 거래 실행
        if action == 0:  # Buy
            quantity = int(risk_metrics.position_size / current_price)
            if quantity > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
                await self.paper_trading.submit_order(order)

                # 손절/익절 주문 자동 설정
                await self.set_stop_orders(symbol, quantity, risk_metrics)

        elif action == 2:  # Sell
            positions = self.paper_trading.get_positions()
            if symbol in positions:
                position = positions[symbol]
                if hasattr(position, 'quantity') and position.quantity > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=position.quantity
                    )
                    await self.paper_trading.submit_order(order)

    async def set_stop_orders(self, symbol: str, quantity: float, risk_metrics):
        """손절/익절 주문 설정"""
        # 손절 주문
        stop_loss_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=risk_metrics.stop_loss
        )
        await self.paper_trading.submit_order(stop_loss_order)

        # 익절 주문
        take_profit_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=risk_metrics.take_profit
        )
        await self.paper_trading.submit_order(take_profit_order)

    def record_daily_performance(self):
        """일일 성과 기록"""
        portfolio_value = self.paper_trading.get_portfolio_value()
        daily_return = (portfolio_value - self.initial_balance) / self.initial_balance

        self.daily_pnl.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': self.paper_trading.cash,
            'positions_value': portfolio_value - self.paper_trading.cash,
            'return': daily_return
        })

    def print_status(self):
        """현재 상태 출력"""
        portfolio_value = self.paper_trading.get_portfolio_value()
        positions = self.paper_trading.get_positions()

        logger.info(f"\n📊 Portfolio Status:")
        logger.info(f"  Total Value: ${portfolio_value:,.2f}")
        logger.info(f"  Cash: ${self.paper_trading.cash:,.2f}")
        logger.info(f"  Return: {((portfolio_value - self.initial_balance) / self.initial_balance):.2%}")

        if positions:
            logger.info(f"\n📈 Positions:")
            for symbol, position in positions.items():
                if hasattr(position, 'quantity'):
                    logger.info(f"  {symbol}: {position.quantity} shares @ ${position.average_cost:.2f}")
                    logger.info(f"    Value: ${position.market_value:,.2f}, P&L: ${position.unrealized_pnl:,.2f}")

    def generate_final_report(self):
        """최종 성과 리포트 생성"""
        metrics = self.paper_trading.get_performance_metrics()
        final_value = self.paper_trading.get_portfolio_value()

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PAPER TRADING FINAL REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance: ${final_value:,.2f}")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Total P&L: ${final_value - self.initial_balance:,.2f}")
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Winning Trades: {metrics.get('winning_trades', 0)}")
        logger.info(f"  Losing Trades: {metrics.get('losing_trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Total Commission: ${self.paper_trading.total_commission:,.2f}")
        logger.info(f"  Total Slippage: ${self.paper_trading.total_slippage:,.2f}")

        # 결과 저장
        self.save_results()

    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('results/paper_trading')
        results_dir.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        self.paper_trading.export_results(
            str(results_dir / f'paper_trading_{timestamp}.json')
        )

        # CSV 저장
        if self.daily_pnl:
            df = pd.DataFrame(self.daily_pnl)
            df.to_csv(results_dir / f'daily_pnl_{timestamp}.csv', index=False)

        logger.info(f"\n✅ Results saved to {results_dir}")


async def main():
    """메인 실행 함수"""
    # 시스템 생성
    system = IntegratedTradingSystem(
        initial_balance=100000,
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    )

    # Paper Trading 실행 (7일)
    await system.run_paper_trading(days=7)

    logger.info("\n✅ Paper Trading simulation completed!")


if __name__ == "__main__":
    asyncio.run(main())