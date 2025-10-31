"""
RL 기반 Paper Trading
학습된 강화학습 모델로 실제 거래 시뮬레이션
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import torch

from implementations.paper_trading_engine import (
    PaperTradingEngine, Order, OrderType, OrderSide
)
from reinforcement_learning.model import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLPaperTrading:
    """
    RL 기반 Paper Trading 시스템
    학습된 DQN 모델로 거래 결정
    """

    def __init__(self, initial_balance: float = 100000, symbols: List[str] = None):
        self.initial_balance = initial_balance
        self.symbols = symbols or ['AAPL']

        # Paper Trading 엔진
        self.engine = PaperTradingEngine(
            initial_balance=initial_balance,
            commission_rate=0.001,
            min_commission=1.0
        )

        # RL 에이전트 (심볼별)
        self.agents: Dict[str, DQNAgent] = {}

        # 시장 데이터
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}

        logger.info(f"RL Paper Trading initialized")
        logger.info(f"Initial balance: ${initial_balance:,.2f}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")

    def load_models(self):
        """학습된 모델 로드"""
        logger.info("Loading trained RL models...")

        for symbol in self.symbols:
            model_path = Path(f'models/{symbol}_paper_trading_model.pth')

            if not model_path.exists():
                logger.warning(f"Model not found for {symbol}: {model_path}")
                logger.warning(f"  Please train the model first: python scripts/quick_train_model.py")
                continue

            try:
                # 에이전트 생성 (epsilon=0 for exploitation only)
                agent = DQNAgent(
                    state_size=23,
                    action_size=3,
                    epsilon=0.0,  # No exploration
                    fc1_units=128,
                    fc2_units=128
                )

                # 모델 로드
                agent.model.load_state_dict(
                    torch.load(model_path, map_location=agent.device, weights_only=True)
                )
                agent.model.eval()

                self.agents[symbol] = agent
                logger.info(f"  ✅ {symbol} model loaded")

            except Exception as e:
                logger.error(f"  ❌ Error loading {symbol} model: {e}")

        if not self.agents:
            raise ValueError("No models loaded! Please train models first.")

        logger.info(f"Loaded {len(self.agents)} models")

    def calculate_features(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """상태 벡터 계산 (23차원)"""
        if len(df) < 20:
            return np.zeros(23)

        close = df['Close'].values
        volume = df['Volume'].values

        # 기본 특징 (3개)
        balance = self.engine.cash
        positions = self.engine.get_positions()
        shares_held = 0

        if symbol in positions:
            pos = positions[symbol]
            if hasattr(pos, 'quantity'):
                shares_held = pos.quantity

        price_norm = close[-1] / close[0] if close[0] > 0 else 1

        features = [
            price_norm,
            balance / self.initial_balance,
            shares_held / 100
        ]

        # 기술적 지표 (20개)
        # SMA
        sma_5 = np.mean(close[-5:])
        sma_10 = np.mean(close[-10:])
        sma_20 = np.mean(close[-20:])

        # RSI
        deltas = np.diff(close[-15:])
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 50

        # Bollinger Bands
        std_20 = np.std(close[-20:])
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # Volatility
        returns = np.diff(close[-20:]) / close[-21:-1]
        volatility = np.std(returns)

        # MACD
        ema_12 = np.mean(close[-12:])
        ema_26 = np.mean(close[-26:]) if len(close) >= 26 else np.mean(close)
        macd = ema_12 - ema_26

        # Momentum
        momentum_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        momentum_10 = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0

        # Volume
        volume_sma = np.mean(volume[-20:])
        volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1

        # 20개 기술적 지표
        tech_features = [
            close[-1] / sma_5 - 1 if sma_5 > 0 else 0,
            close[-1] / sma_10 - 1 if sma_10 > 0 else 0,
            close[-1] / sma_20 - 1 if sma_20 > 0 else 0,
            rsi / 100,
            bb_position,
            volatility,
            macd / close[-1] if close[-1] > 0 else 0,
            momentum_5,
            momentum_10,
            np.log1p(volume_ratio),
            (close[-1] - np.min(close[-20:])) / (np.max(close[-20:]) - np.min(close[-20:])) if (np.max(close[-20:]) - np.min(close[-20:])) > 0 else 0.5,
            np.mean(returns),
            np.max(returns) if len(returns) > 0 else 0,
            np.min(returns) if len(returns) > 0 else 0,
            (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0,
            std_20 / close[-1] if close[-1] > 0 else 0,
            (bb_upper - bb_lower) / close[-1] if close[-1] > 0 else 0,
            (close[-1] - close[-5]) / close[-5] if len(close) >= 5 and close[-5] > 0 else 0,
            (volume[-1] - volume[-5]) / volume[-5] if len(volume) >= 5 and volume[-5] > 0 else 0,
            np.max(close[-20:]) / close[-1] - 1 if close[-1] > 0 else 0,
        ]

        features.extend(tech_features)
        return np.array(features[:23])

    def load_data(self, days: int = 30):
        """시장 데이터 로드"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Loading {days} days of market data...")

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if not df.empty:
                    self.market_data[symbol] = df
                    self.current_prices[symbol] = df['Close'].iloc[-1]
                    logger.info(f"  {symbol}: {len(df)} days, price: ${df['Close'].iloc[-1]:.2f}")

            except Exception as e:
                logger.error(f"  {symbol}: Error - {e}")

    async def run(self, days: int = 7):
        """RL Paper Trading 실행"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 RL-Based Paper Trading Simulation ({days} days)")
        logger.info(f"{'='*70}\n")

        # 모델 로드
        self.load_models()

        # 데이터 로드
        self.load_data(days=days + 30)  # 기술적 지표 계산을 위한 추가 데이터

        if not self.market_data:
            logger.error("No market data available")
            return

        # 일별 시뮬레이션
        for day in range(days):
            logger.info(f"\n📅 Day {day + 1}/{days}")
            logger.info("-" * 70)

            for symbol in self.symbols:
                if symbol not in self.market_data or symbol not in self.agents:
                    continue

                # 현재까지의 데이터
                df = self.market_data[symbol]
                if len(df) > day + 30:
                    current_df = df.iloc[:-(days - day)]

                    if len(current_df) >= 30:
                        # 특징 계산
                        features = self.calculate_features(symbol, current_df)

                        # RL 모델 예측
                        agent = self.agents[symbol]
                        action = agent.act(features.reshape(1, -1))

                        # 현재 가격
                        current_price = current_df['Close'].iloc[-1]
                        self.current_prices[symbol] = current_price

                        # 거래 실행
                        await self.execute_action(symbol, action, current_price)

            # 가격 업데이트
            await self.engine.update_market_prices(self.current_prices)

            # 일일 요약
            self.print_daily_summary(day + 1)

            await asyncio.sleep(0.1)

        # 최종 결과
        self.print_final_report()

    async def execute_action(self, symbol: str, action: int, price: float):
        """RL 모델 액션 실행"""
        portfolio_value = self.engine.get_portfolio_value()
        positions = self.engine.get_positions()

        if action == 0:  # Buy
            # 포트폴리오의 20%로 매수
            buy_amount = portfolio_value * 0.20
            quantity = int(buy_amount / price)

            if quantity > 0 and self.engine.cash >= quantity * price:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
                success = await self.engine.submit_order(order)
                if success:
                    logger.info(f"  🟢 BUY  {symbol}: {quantity} shares @ ${price:.2f} (RL Decision)")

        elif action == 2:  # Sell
            if symbol in positions:
                pos = positions[symbol]
                if hasattr(pos, 'quantity') and pos.quantity > 0:
                    # 전량 매도
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=pos.quantity
                    )
                    success = await self.engine.submit_order(order)
                    if success:
                        logger.info(f"  🔴 SELL {symbol}: {pos.quantity} shares @ ${price:.2f} (RL Decision)")

    def print_daily_summary(self, day: int):
        """일일 요약"""
        portfolio_value = self.engine.get_portfolio_value()
        positions = self.engine.get_positions()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance

        logger.info(f"\n📊 Day {day} Summary:")
        logger.info(f"  Portfolio: ${portfolio_value:,.2f} | Return: {total_return:+.2%}")

        if positions:
            logger.info(f"  Positions: {len(positions)}")
            for symbol, pos in list(positions.items())[:3]:  # 최대 3개만 표시
                if hasattr(pos, 'unrealized_pnl'):
                    logger.info(f"    {symbol}: ${pos.unrealized_pnl:+,.2f}")

    def print_final_report(self):
        """최종 보고서"""
        metrics = self.engine.get_performance_metrics()
        final_value = self.engine.get_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance

        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 RL PAPER TRADING FINAL RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"\n💰 Performance:")
        logger.info(f"  Initial:      ${self.initial_balance:>12,.2f}")
        logger.info(f"  Final:        ${final_value:>12,.2f}")
        logger.info(f"  P&L:          ${final_value - self.initial_balance:>+12,.2f}")
        logger.info(f"  Return:       {total_return:>12.2%}")

        logger.info(f"\n📈 Statistics:")
        logger.info(f"  Trades:       {metrics.get('total_trades', 0):>12}")
        logger.info(f"  Win Rate:     {metrics.get('win_rate', 0):>12.1%}")
        logger.info(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):>12.2f}")
        logger.info(f"  Max DD:       {metrics.get('max_drawdown', 0):>12.1%}")

        logger.info(f"\n💸 Costs:")
        logger.info(f"  Commission:   ${self.engine.total_commission:>12,.2f}")
        logger.info(f"  Slippage:     ${self.engine.total_slippage:>12,.2f}")

        # 저장
        results_dir = Path('results/paper_trading')
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.engine.export_results(str(results_dir / f'rl_paper_trading_{timestamp}.json'))

        logger.info(f"\n💾 Results saved to: results/paper_trading/")
        logger.info(f"\n{'='*70}")


async def main():
    """메인 실행"""
    print("\n" + "="*70)
    print("🤖 InvenstX RL Paper Trading")
    print("="*70)
    print("\n학습된 강화학습 모델로 Paper Trading을 실행합니다.\n")

    config = {
        'initial_balance': 100000,
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'days': 7
    }

    print(f"📋 Configuration:")
    print(f"  Initial Balance: ${config['initial_balance']:,.2f}")
    print(f"  Symbols: {', '.join(config['symbols'])}")
    print(f"  Days: {config['days']}")

    response = input("\n시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ 취소")
        return

    system = RLPaperTrading(
        initial_balance=config['initial_balance'],
        symbols=config['symbols']
    )

    try:
        await system.run(days=config['days'])
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n먼저 모델을 학습하세요:")
        print("  python scripts/quick_train_model.py")


if __name__ == "__main__":
    asyncio.run(main())