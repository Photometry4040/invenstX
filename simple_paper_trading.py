"""
간단한 Paper Trading 테스트
ML 모델 없이 랜덤 전략으로 시스템 검증
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json

from implementations.paper_trading_engine import (
    PaperTradingEngine, Order, OrderType, OrderSide
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePaperTrading:
    """
    간단한 Paper Trading 시스템
    랜덤 전략으로 시스템 검증
    """

    def __init__(self, initial_balance: float = 100000, symbols: List[str] = None):
        self.initial_balance = initial_balance
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT']

        # Paper Trading 엔진
        self.engine = PaperTradingEngine(
            initial_balance=initial_balance,
            commission_rate=0.001,
            min_commission=1.0
        )

        # 시장 데이터
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}

        logger.info(f"Simple Paper Trading initialized")
        logger.info(f"Initial balance: ${initial_balance:,.2f}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")

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
                    logger.info(f"  {symbol}: {len(df)} days loaded, current price: ${df['Close'].iloc[-1]:.2f}")
                else:
                    logger.warning(f"  {symbol}: No data available")

            except Exception as e:
                logger.error(f"  {symbol}: Error loading data - {e}")

    def random_strategy(self, symbol: str) -> int:
        """
        랜덤 전략
        Returns:
            0: Buy, 1: Hold, 2: Sell
        """
        # 70% Hold, 15% Buy, 15% Sell
        return np.random.choice([0, 1, 2], p=[0.15, 0.70, 0.15])

    async def run(self, days: int = 7):
        """Paper Trading 실행"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting {days}-day Paper Trading Simulation")
        logger.info(f"{'='*70}\n")

        # 데이터 로드
        self.load_data(days=days + 20)

        if not self.market_data:
            logger.error("No market data available. Exiting.")
            return

        # 일별 시뮬레이션
        for day in range(days):
            logger.info(f"\n📅 Day {day + 1}/{days}")
            logger.info("-" * 70)

            # 각 종목에 대해 거래 결정
            for symbol in self.symbols:
                if symbol not in self.market_data:
                    continue

                # 현재 가격 (시뮬레이션을 위해 과거 데이터 사용)
                df = self.market_data[symbol]
                if len(df) > day:
                    current_price = df['Close'].iloc[-(days - day)]
                    self.current_prices[symbol] = current_price

                    # 랜덤 전략으로 거래 결정
                    action = self.random_strategy(symbol)

                    await self.execute_action(symbol, action, current_price)

            # 가격 업데이트
            await self.engine.update_market_prices(self.current_prices)

            # 일일 요약
            self.print_daily_summary(day + 1)

            await asyncio.sleep(0.1)

        # 최종 결과
        self.print_final_report()

    async def execute_action(self, symbol: str, action: int, price: float):
        """거래 실행"""
        portfolio_value = self.engine.get_portfolio_value()
        positions = self.engine.get_positions()

        if action == 0:  # Buy
            # 포트폴리오의 10%로 매수
            buy_amount = portfolio_value * 0.10
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
                    logger.info(f"  ✅ BUY  {symbol}: {quantity} shares @ ${price:.2f}")

        elif action == 2:  # Sell
            # 보유 중인 포지션이 있으면 절반 매도
            if symbol in positions:
                position = positions[symbol]
                if hasattr(position, 'quantity') and position.quantity > 0:
                    sell_quantity = max(1, int(position.quantity * 0.5))
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=sell_quantity
                    )
                    success = await self.engine.submit_order(order)
                    if success:
                        logger.info(f"  ✅ SELL {symbol}: {sell_quantity} shares @ ${price:.2f}")

        # Hold는 아무것도 하지 않음

    def print_daily_summary(self, day: int):
        """일일 요약"""
        portfolio_value = self.engine.get_portfolio_value()
        positions = self.engine.get_positions()

        total_return = (portfolio_value - self.initial_balance) / self.initial_balance

        logger.info(f"\n📊 End of Day {day} Summary:")
        logger.info(f"  Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"  Cash: ${self.engine.cash:,.2f}")
        logger.info(f"  Total Return: {total_return:+.2%}")

        if positions:
            logger.info(f"  Active Positions: {len(positions)}")
            for symbol, pos in positions.items():
                if hasattr(pos, 'quantity'):
                    logger.info(f"    {symbol}: {pos.quantity} shares, P&L: ${pos.unrealized_pnl:+,.2f}")

    def print_final_report(self):
        """최종 보고서"""
        metrics = self.engine.get_performance_metrics()
        final_value = self.engine.get_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance

        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FINAL RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"\n💰 Financial Performance:")
        logger.info(f"  Initial Balance:  ${self.initial_balance:>12,.2f}")
        logger.info(f"  Final Balance:    ${final_value:>12,.2f}")
        logger.info(f"  Total P&L:        ${final_value - self.initial_balance:>+12,.2f}")
        logger.info(f"  Total Return:     {total_return:>12.2%}")

        logger.info(f"\n📈 Trading Statistics:")
        logger.info(f"  Total Trades:     {metrics.get('total_trades', 0):>12}")
        logger.info(f"  Winning Trades:   {metrics.get('winning_trades', 0):>12}")
        logger.info(f"  Losing Trades:    {metrics.get('losing_trades', 0):>12}")
        logger.info(f"  Win Rate:         {metrics.get('win_rate', 0):>12.1%}")

        logger.info(f"\n📉 Risk Metrics:")
        logger.info(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>12.2f}")
        logger.info(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):>12.1%}")

        logger.info(f"\n💸 Costs:")
        logger.info(f"  Total Commission: ${self.engine.total_commission:>12,.2f}")
        logger.info(f"  Total Slippage:   ${self.engine.total_slippage:>12,.2f}")

        # 포지션 상태
        positions = self.engine.get_positions()
        if positions:
            logger.info(f"\n📦 Final Positions:")
            for symbol, pos in positions.items():
                if hasattr(pos, 'quantity'):
                    logger.info(f"  {symbol}:")
                    logger.info(f"    Quantity:       {pos.quantity:>12}")
                    logger.info(f"    Avg Cost:       ${pos.average_cost:>12.2f}")
                    logger.info(f"    Current Price:  ${pos.current_price:>12.2f}")
                    logger.info(f"    Market Value:   ${pos.market_value:>12,.2f}")
                    logger.info(f"    Unrealized P&L: ${pos.unrealized_pnl:>+12,.2f}")

        # 결과 저장
        self.save_results()

        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Paper Trading Simulation Completed!")
        logger.info(f"{'='*70}\n")

    def save_results(self):
        """결과 저장"""
        results_dir = Path('results/paper_trading')
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = results_dir / f'simple_paper_trading_{timestamp}.json'

        self.engine.export_results(str(filepath))
        logger.info(f"\n💾 Results saved to: {filepath}")


async def main():
    """메인 실행"""
    print("\n" + "="*70)
    print("🎯 InvenstX Simple Paper Trading Test")
    print("="*70)
    print("\n이 테스트는 ML 모델 없이 랜덤 전략으로 시스템을 검증합니다.")
    print("실제 Paper Trading 엔진의 작동을 확인할 수 있습니다.\n")

    # 설정
    config = {
        'initial_balance': 100000,
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'days': 7
    }

    print(f"📋 Configuration:")
    print(f"  Initial Balance: ${config['initial_balance']:,.2f}")
    print(f"  Symbols: {', '.join(config['symbols'])}")
    print(f"  Trading Days: {config['days']}")

    response = input("\n시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ 취소되었습니다.")
        return

    # Paper Trading 실행
    system = SimplePaperTrading(
        initial_balance=config['initial_balance'],
        symbols=config['symbols']
    )

    await system.run(days=config['days'])


if __name__ == "__main__":
    asyncio.run(main())