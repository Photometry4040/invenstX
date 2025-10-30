"""
Paper Trading Engine
실제 시장 조건을 시뮬레이션하는 가상 거래 엔진
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """주문 클래스"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    average_fill_price: float = 0
    commission: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class Position:
    """포지션 클래스"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float = 0
    market_value: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_price(self, price: float):
        """가격 업데이트 및 손익 계산"""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.average_cost) * self.quantity
        self.updated_at = datetime.now()


@dataclass
class Trade:
    """체결 기록"""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0
    price: float = 0
    commission: float = 0
    executed_at: datetime = field(default_factory=datetime.now)
    slippage: float = 0
    notes: str = ""


class MarketSimulator:
    """시장 시뮬레이터 - 실제 시장 조건 모사"""

    def __init__(self,
                 bid_ask_spread: float = 0.01,
                 slippage_factor: float = 0.001,
                 market_impact_factor: float = 0.0001,
                 fill_probability: float = 0.95):
        """
        Args:
            bid_ask_spread: 평균 스프레드 (%)
            slippage_factor: 슬리피지 계수
            market_impact_factor: 시장 충격 계수
            fill_probability: 지정가 주문 체결 확률
        """
        self.bid_ask_spread = bid_ask_spread
        self.slippage_factor = slippage_factor
        self.market_impact_factor = market_impact_factor
        self.fill_probability = fill_probability

    def get_execution_price(self,
                           current_price: float,
                           side: OrderSide,
                           quantity: float,
                           order_type: OrderType) -> Tuple[float, float]:
        """
        실제 체결 가격 계산

        Returns:
            (체결가격, 슬리피지)
        """
        # 스프레드 계산
        half_spread = current_price * self.bid_ask_spread / 200

        if side == OrderSide.BUY:
            # 매수는 Ask 가격
            base_price = current_price + half_spread
        else:
            # 매도는 Bid 가격
            base_price = current_price - half_spread

        # 슬리피지 계산 (수량에 비례)
        slippage = base_price * self.slippage_factor * np.log1p(quantity / 100)

        # 시장 충격 (큰 주문일수록 가격에 영향)
        market_impact = base_price * self.market_impact_factor * quantity

        # 랜덤 요소 추가
        random_factor = np.random.normal(0, 0.0001)

        if side == OrderSide.BUY:
            execution_price = base_price + slippage + market_impact + base_price * random_factor
        else:
            execution_price = base_price - slippage - market_impact + base_price * random_factor

        actual_slippage = abs(execution_price - current_price) / current_price

        return execution_price, actual_slippage

    def should_fill_limit_order(self,
                               limit_price: float,
                               current_price: float,
                               side: OrderSide) -> bool:
        """지정가 주문 체결 여부 결정"""
        if side == OrderSide.BUY:
            # 매수 지정가는 현재가가 지정가 이하일 때
            if current_price <= limit_price:
                return np.random.random() < self.fill_probability
        else:
            # 매도 지정가는 현재가가 지정가 이상일 때
            if current_price >= limit_price:
                return np.random.random() < self.fill_probability
        return False


class PaperTradingEngine:
    """
    Paper Trading 엔진
    실제 거래를 시뮬레이션하는 가상 거래 시스템
    """

    def __init__(self,
                 initial_balance: float = 100000,
                 commission_rate: float = 0.001,
                 min_commission: float = 1.0,
                 margin_enabled: bool = False,
                 margin_rate: float = 2.0):
        """
        Args:
            initial_balance: 초기 자본
            commission_rate: 수수료율
            min_commission: 최소 수수료
            margin_enabled: 마진 거래 활성화
            margin_rate: 마진 배율
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.margin_enabled = margin_enabled
        self.margin_rate = margin_rate

        # 포지션 및 주문 관리
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.trades: List[Trade] = []

        # 성과 추적
        self.equity_curve = [initial_balance]
        self.daily_returns = []
        self.total_commission = 0
        self.total_slippage = 0

        # 시장 시뮬레이터
        self.market_simulator = MarketSimulator()

        # 실시간 가격 데이터
        self.current_prices: Dict[str, float] = {}

        # 리스크 한계
        self.max_position_size = 0.1  # 포트폴리오의 10%
        self.max_daily_loss = 0.02     # 일일 최대 손실 2%
        self.max_positions = 10        # 최대 동시 포지션

        # 성과 메트릭
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }

    async def submit_order(self, order: Order) -> bool:
        """
        주문 제출

        Args:
            order: 주문 객체

        Returns:
            성공 여부
        """
        # 주문 검증
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order.order_id}")
            return False

        # 리스크 체크
        if not self._check_risk_limits(order):
            order.status = OrderStatus.REJECTED
            order.notes = "Risk limit exceeded"
            logger.warning(f"Order rejected due to risk limits: {order.order_id}")
            return False

        # 주문 등록
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()
        self.orders[order.order_id] = order

        if order.order_type == OrderType.MARKET:
            # 시장가 주문은 즉시 체결
            await self._execute_market_order(order)
        else:
            # 지정가/스톱 주문은 대기 목록에 추가
            self.pending_orders.append(order)

        logger.info(f"Order submitted: {order.order_id} - {order.symbol} {order.side.value} {order.quantity}")
        return True

    async def _execute_market_order(self, order: Order):
        """시장가 주문 체결"""
        current_price = self.current_prices.get(order.symbol, 0)
        if current_price == 0:
            order.status = OrderStatus.REJECTED
            order.notes = "No market data available"
            return

        # 체결 가격 계산 (슬리피지 포함)
        execution_price, slippage = self.market_simulator.get_execution_price(
            current_price, order.side, order.quantity, order.order_type
        )

        # 수수료 계산
        commission = self._calculate_commission(order.quantity * execution_price)

        # 잔고 확인
        total_cost = order.quantity * execution_price + commission
        if order.side == OrderSide.BUY and total_cost > self.cash:
            order.status = OrderStatus.REJECTED
            order.notes = "Insufficient funds"
            return

        # 체결 처리
        self._process_fill(order, order.quantity, execution_price, commission, slippage)

    def _process_fill(self,
                      order: Order,
                      filled_quantity: float,
                      fill_price: float,
                      commission: float,
                      slippage: float):
        """주문 체결 처리"""
        # 주문 업데이트
        order.filled_quantity += filled_quantity
        order.average_fill_price = fill_price
        order.commission += commission
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()

        # Trade 기록 생성
        trade = Trade(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=filled_quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage
        )
        self.trades.append(trade)

        # 포지션 업데이트
        self._update_position(order.symbol, order.side, filled_quantity, fill_price)

        # 잔고 업데이트
        if order.side == OrderSide.BUY:
            self.cash -= (filled_quantity * fill_price + commission)
        else:
            self.cash += (filled_quantity * fill_price - commission)

        # 통계 업데이트
        self.total_commission += commission
        self.total_slippage += slippage * filled_quantity * fill_price
        self.metrics['total_trades'] += 1

        logger.info(f"Order filled: {order.order_id} - {filled_quantity} @ {fill_price:.2f}")

    def _update_position(self,
                        symbol: str,
                        side: OrderSide,
                        quantity: float,
                        price: float):
        """포지션 업데이트"""
        if symbol not in self.positions:
            if side == OrderSide.BUY:
                # 새 포지션 생성
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_cost=price,
                    current_price=price
                )
        else:
            position = self.positions[symbol]
            if side == OrderSide.BUY:
                # 포지션 추가
                total_cost = position.quantity * position.average_cost + quantity * price
                position.quantity += quantity
                position.average_cost = total_cost / position.quantity
            else:
                # 포지션 감소
                if quantity >= position.quantity:
                    # 포지션 청산
                    realized_pnl = (price - position.average_cost) * position.quantity
                    position.realized_pnl += realized_pnl
                    del self.positions[symbol]

                    # 성과 업데이트
                    self.metrics['total_pnl'] += realized_pnl
                    if realized_pnl > 0:
                        self.metrics['winning_trades'] += 1
                    else:
                        self.metrics['losing_trades'] += 1
                else:
                    # 부분 청산
                    realized_pnl = (price - position.average_cost) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    self.metrics['total_pnl'] += realized_pnl

    async def update_market_prices(self, prices: Dict[str, float]):
        """시장 가격 업데이트"""
        self.current_prices.update(prices)

        # 포지션 가치 업데이트
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        # 대기 중인 지정가/스톱 주문 체크
        await self._check_pending_orders()

        # 포트폴리오 가치 계산
        self._update_portfolio_value()

    async def _check_pending_orders(self):
        """대기 중인 주문 체크 및 체결"""
        filled_orders = []

        for order in self.pending_orders:
            if order.symbol not in self.current_prices:
                continue

            current_price = self.current_prices[order.symbol]

            # 지정가 주문 체크
            if order.order_type == OrderType.LIMIT:
                if self.market_simulator.should_fill_limit_order(
                    order.price, current_price, order.side
                ):
                    execution_price = order.price
                    commission = self._calculate_commission(order.quantity * execution_price)
                    self._process_fill(order, order.quantity, execution_price, commission, 0)
                    filled_orders.append(order)

            # 스톱 주문 체크
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    # 매수 스톱
                    await self._execute_market_order(order)
                    filled_orders.append(order)
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    # 매도 스톱 (손절)
                    await self._execute_market_order(order)
                    filled_orders.append(order)

        # 체결된 주문 제거
        for order in filled_orders:
            self.pending_orders.remove(order)

    def _validate_order(self, order: Order) -> bool:
        """주문 유효성 검증"""
        if order.quantity <= 0:
            return False
        if order.symbol == "":
            return False
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False
        if order.order_type == OrderType.STOP and order.stop_price is None:
            return False
        return True

    def _check_risk_limits(self, order: Order) -> bool:
        """리스크 한계 체크"""
        # 포지션 수 제한
        if len(self.positions) >= self.max_positions and order.symbol not in self.positions:
            return False

        # 포지션 크기 제한
        portfolio_value = self.get_portfolio_value()
        order_value = order.quantity * self.current_prices.get(order.symbol, 0)
        if order_value > portfolio_value * self.max_position_size:
            return False

        # 일일 손실 제한
        daily_loss = (self.initial_balance - portfolio_value) / self.initial_balance
        if daily_loss > self.max_daily_loss:
            return False

        return True

    def _calculate_commission(self, trade_value: float) -> float:
        """수수료 계산"""
        return max(self.min_commission, trade_value * self.commission_rate)

    def _update_portfolio_value(self):
        """포트폴리오 가치 업데이트"""
        portfolio_value = self.cash
        for position in self.positions.values():
            portfolio_value += position.market_value

        self.equity_curve.append(portfolio_value)

        # 최대 드로다운 계산
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            drawdown = (peak - portfolio_value) / peak
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)

    def get_portfolio_value(self) -> float:
        """현재 포트폴리오 가치"""
        portfolio_value = self.cash
        for position in self.positions.values():
            portfolio_value += position.market_value
        return portfolio_value

    def get_positions(self) -> Dict[str, Position]:
        """현재 포지션 반환"""
        return self.positions.copy()

    def get_performance_metrics(self) -> Dict:
        """성과 메트릭 반환"""
        # 승률 계산
        total_trades = self.metrics['winning_trades'] + self.metrics['losing_trades']
        if total_trades > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / total_trades

        # Sharpe Ratio 계산
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # 총 수익률
        current_value = self.get_portfolio_value()
        self.metrics['total_return'] = (current_value - self.initial_balance) / self.initial_balance

        return self.metrics.copy()

    def get_trade_history(self) -> List[Trade]:
        """거래 내역 반환"""
        return self.trades.copy()

    def reset(self):
        """엔진 초기화"""
        self.cash = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        self.trades.clear()
        self.equity_curve = [self.initial_balance]
        self.daily_returns.clear()
        self.total_commission = 0
        self.total_slippage = 0
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }
        logger.info("Paper trading engine reset")

    def export_results(self, filepath: str):
        """결과 내보내기"""
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': self.get_portfolio_value(),
            'metrics': self.get_performance_metrics(),
            'positions': [asdict(pos) for pos in self.positions.values()],
            'trades': [asdict(trade) for trade in self.trades],
            'equity_curve': self.equity_curve,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results exported to {filepath}")


async def main():
    """테스트 함수"""

    # Paper Trading 엔진 생성
    engine = PaperTradingEngine(initial_balance=100000)

    # 샘플 가격 데이터
    prices = {
        'AAPL': 150.00,
        'GOOGL': 140.00,
        'TSLA': 240.00
    }
    await engine.update_market_prices(prices)

    # 매수 주문
    buy_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    await engine.submit_order(buy_order)

    # 지정가 매도 주문
    sell_order = Order(
        symbol='AAPL',
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50,
        price=155.00
    )
    await engine.submit_order(sell_order)

    # 가격 업데이트
    await asyncio.sleep(1)
    prices['AAPL'] = 156.00
    await engine.update_market_prices(prices)

    # 손절 주문
    stop_loss_order = Order(
        symbol='AAPL',
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=50,
        stop_price=145.00
    )
    await engine.submit_order(stop_loss_order)

    # 성과 출력
    print("\n=== Portfolio Status ===")
    print(f"Cash: ${engine.cash:,.2f}")
    print(f"Portfolio Value: ${engine.get_portfolio_value():,.2f}")

    print("\n=== Positions ===")
    for symbol, position in engine.get_positions().items():
        print(f"{symbol}: {position.quantity} shares @ ${position.average_cost:.2f}")
        print(f"  Market Value: ${position.market_value:,.2f}")
        print(f"  Unrealized P&L: ${position.unrealized_pnl:,.2f}")

    print("\n=== Performance Metrics ===")
    metrics = engine.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key or 'ratio' in key or 'return' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # 결과 저장
    engine.export_results('paper_trading_results.json')


if __name__ == "__main__":
    asyncio.run(main())