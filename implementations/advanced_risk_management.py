"""
고급 리스크 관리 시스템
Kelly Criterion, VaR, CVaR, 동적 손절/익절 등 포함
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """리스크 메트릭"""
    position_size: float
    stop_loss: float
    take_profit: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction: float


class KellyCriterion:
    """
    Kelly Criterion 포지션 사이징
    최적 베팅 크기 계산
    """

    def __init__(self, max_fraction: float = 0.25):
        """
        Args:
            max_fraction: 최대 Kelly 비율 (안전성을 위한 상한)
        """
        self.max_fraction = max_fraction
        self.historical_results = []

    def calculate(self,
                  win_probability: float,
                  win_loss_ratio: float,
                  confidence: float = 1.0) -> float:
        """
        Kelly Criterion 계산

        f* = (bp - q) / b
        where:
            f* = 투자 비율
            b = 승리 시 수익 배수
            p = 승리 확률
            q = 패배 확률 (1-p)

        Args:
            win_probability: 승리 확률 (0-1)
            win_loss_ratio: 평균 수익/평균 손실 비율
            confidence: 신뢰도 조정 계수 (0-1)

        Returns:
            최적 포지션 크기 비율
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0

        if win_loss_ratio <= 0:
            return 0

        q = 1 - win_probability
        kelly_fraction = (win_loss_ratio * win_probability - q) / win_loss_ratio

        # 음수 Kelly는 베팅하지 않음을 의미
        if kelly_fraction < 0:
            return 0

        # 신뢰도 조정
        kelly_fraction *= confidence

        # 상한 적용 (보수적 접근)
        kelly_fraction = min(kelly_fraction, self.max_fraction)

        logger.info(f"Kelly Fraction: {kelly_fraction:.4f} (p={win_probability:.2f}, b={win_loss_ratio:.2f})")

        return kelly_fraction

    def calculate_from_history(self, returns: np.ndarray) -> float:
        """
        과거 수익률 데이터로부터 Kelly Criterion 계산

        Args:
            returns: 과거 수익률 배열

        Returns:
            최적 포지션 크기 비율
        """
        if len(returns) < 10:
            logger.warning("Insufficient data for Kelly calculation")
            return 0.02  # 기본값

        # 승리/패배 분석
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.02  # 기본값

        win_probability = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return self.max_fraction

        win_loss_ratio = avg_win / avg_loss

        return self.calculate(win_probability, win_loss_ratio)


class ValueAtRisk:
    """
    Value at Risk (VaR) 및 Conditional VaR (CVaR) 계산
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: 신뢰 수준 (예: 0.95 = 95%)
        """
        self.confidence_level = confidence_level

    def calculate_var(self, returns: np.ndarray, method: str = 'historical') -> float:
        """
        VaR 계산

        Args:
            returns: 수익률 데이터
            method: 'historical', 'parametric', 'montecarlo'

        Returns:
            VaR 값 (손실은 음수)
        """
        if len(returns) == 0:
            return 0

        if method == 'historical':
            # Historical VaR
            var = np.percentile(returns, (1 - self.confidence_level) * 100)

        elif method == 'parametric':
            # Parametric VaR (정규분포 가정)
            mean = np.mean(returns)
            std = np.std(returns)
            var = mean + std * stats.norm.ppf(1 - self.confidence_level)

        elif method == 'montecarlo':
            # Monte Carlo VaR
            var = self._monte_carlo_var(returns)

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"VaR ({self.confidence_level*100}%, {method}): {var:.4f}")

        return var

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """
        Conditional VaR (Expected Shortfall) 계산

        Args:
            returns: 수익률 데이터

        Returns:
            CVaR 값
        """
        if len(returns) == 0:
            return 0

        var = self.calculate_var(returns)
        # VaR을 초과하는 손실들의 평균
        cvar = np.mean(returns[returns <= var])

        if np.isnan(cvar):
            cvar = var

        logger.info(f"CVaR ({self.confidence_level*100}%): {cvar:.4f}")

        return cvar

    def _monte_carlo_var(self, returns: np.ndarray, simulations: int = 10000) -> float:
        """Monte Carlo 시뮬레이션을 통한 VaR 계산"""
        mean = np.mean(returns)
        std = np.std(returns)

        # 시뮬레이션
        simulated_returns = np.random.normal(mean, std, simulations)
        var = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)

        return var


class DynamicStopLoss:
    """
    동적 손절/익절 관리
    ATR, 추세 강도, 변동성 기반
    """

    def __init__(self,
                 atr_multiplier: float = 2.0,
                 trailing_stop_pct: float = 0.05):
        """
        Args:
            atr_multiplier: ATR 배수 (손절 거리)
            trailing_stop_pct: 추적 손절 비율
        """
        self.atr_multiplier = atr_multiplier
        self.trailing_stop_pct = trailing_stop_pct

    def calculate_stops(self,
                       entry_price: float,
                       atr: float,
                       volatility: float,
                       trend_strength: float = 0.5,
                       position_type: str = 'long') -> Dict:
        """
        동적 손절/익절 계산

        Args:
            entry_price: 진입 가격
            atr: Average True Range
            volatility: 변동성 (표준편차)
            trend_strength: 추세 강도 (-1 to 1)
            position_type: 'long' or 'short'

        Returns:
            손절/익절 가격 딕셔너리
        """
        # 기본 손절 거리
        stop_distance = atr * self.atr_multiplier

        # 변동성 조정
        volatility_adjustment = 1 + (volatility - 0.15) * 2  # 15%를 기준으로 조정
        stop_distance *= max(0.5, min(2.0, volatility_adjustment))

        # 추세 강도 조정
        trend_adjustment = 1 - abs(trend_strength) * 0.3  # 강한 추세일수록 타이트한 손절
        stop_distance *= trend_adjustment

        if position_type == 'long':
            stop_loss = entry_price - stop_distance
            # 익절은 손절의 1.5-3배 (추세 강도에 따라)
            take_profit_multiplier = 1.5 + max(0, trend_strength) * 1.5
            take_profit = entry_price + stop_distance * take_profit_multiplier

        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit_multiplier = 1.5 + max(0, -trend_strength) * 1.5
            take_profit = entry_price - stop_distance * take_profit_multiplier

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_pct': self.trailing_stop_pct,
            'stop_distance': stop_distance,
            'risk_reward_ratio': take_profit_multiplier
        }

    def update_trailing_stop(self,
                            current_price: float,
                            entry_price: float,
                            current_stop: float,
                            position_type: str = 'long') -> float:
        """
        추적 손절 업데이트

        Args:
            current_price: 현재 가격
            entry_price: 진입 가격
            current_stop: 현재 손절 가격
            position_type: 'long' or 'short'

        Returns:
            업데이트된 손절 가격
        """
        if position_type == 'long':
            # 가격이 올라갔을 때만 손절 상향
            if current_price > entry_price:
                new_stop = current_price * (1 - self.trailing_stop_pct)
                return max(current_stop, new_stop)
        else:  # short
            # 가격이 내려갔을 때만 손절 하향
            if current_price < entry_price:
                new_stop = current_price * (1 + self.trailing_stop_pct)
                return min(current_stop, new_stop)

        return current_stop


class PortfolioRiskManager:
    """
    포트폴리오 전체 리스크 관리
    """

    def __init__(self,
                 max_portfolio_risk: float = 0.06,  # 6% 최대 포트폴리오 리스크
                 max_correlation: float = 0.7,      # 최대 상관관계
                 max_concentration: float = 0.2):   # 최대 집중도
        """
        Args:
            max_portfolio_risk: 최대 포트폴리오 리스크 (VaR 기준)
            max_correlation: 포지션 간 최대 상관관계
            max_concentration: 단일 포지션 최대 비중
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration

        self.kelly = KellyCriterion()
        self.var_calculator = ValueAtRisk()
        self.stop_loss_manager = DynamicStopLoss()

    def calculate_position_size(self,
                               symbol: str,
                               confidence: float,
                               returns_history: np.ndarray,
                               portfolio_value: float,
                               current_positions: Dict) -> float:
        """
        포지션 크기 계산 (모든 리스크 요소 고려)

        Args:
            symbol: 종목 심볼
            confidence: 신호 신뢰도
            returns_history: 과거 수익률
            portfolio_value: 포트폴리오 가치
            current_positions: 현재 포지션

        Returns:
            포지션 크기 (달러)
        """
        # Kelly Criterion
        kelly_fraction = self.kelly.calculate_from_history(returns_history)

        # VaR 제약
        var = abs(self.var_calculator.calculate_var(returns_history))
        var_limit = self.max_portfolio_risk / var if var > 0 else 1.0

        # 집중도 제약
        concentration_limit = self.max_concentration

        # 상관관계 조정 (현재 포지션과의 상관관계 고려)
        correlation_adjustment = self._calculate_correlation_adjustment(
            symbol, current_positions
        )

        # 최종 포지션 크기
        position_fraction = min(
            kelly_fraction * confidence,
            var_limit,
            concentration_limit
        ) * correlation_adjustment

        position_size = portfolio_value * position_fraction

        logger.info(f"Position size for {symbol}: ${position_size:.2f} ({position_fraction:.2%})")

        return position_size

    def _calculate_correlation_adjustment(self,
                                         symbol: str,
                                         current_positions: Dict) -> float:
        """
        상관관계 기반 포지션 크기 조정

        Returns:
            조정 계수 (0-1)
        """
        if not current_positions:
            return 1.0

        # 단순화된 상관관계 체크
        # 실제로는 히스토리컬 상관관계 계산 필요
        similar_positions = sum(1 for pos in current_positions.values()
                              if pos.get('sector') == symbol[:2])  # 섹터 기반 단순 체크

        if similar_positions > 0:
            return max(0.5, 1.0 - similar_positions * 0.2)

        return 1.0

    def check_risk_limits(self, portfolio: Dict) -> Dict[str, bool]:
        """
        포트폴리오 리스크 한계 체크

        Args:
            portfolio: 포트폴리오 데이터

        Returns:
            리스크 체크 결과
        """
        checks = {
            'portfolio_var': True,
            'concentration': True,
            'correlation': True,
            'max_drawdown': True
        }

        # 포트폴리오 VaR 체크
        if 'returns' in portfolio:
            portfolio_var = abs(self.var_calculator.calculate_var(portfolio['returns']))
            checks['portfolio_var'] = portfolio_var <= self.max_portfolio_risk

        # 집중도 체크
        if 'positions' in portfolio:
            total_value = sum(pos['value'] for pos in portfolio['positions'].values())
            for pos in portfolio['positions'].values():
                if pos['value'] / total_value > self.max_concentration:
                    checks['concentration'] = False
                    break

        # 최대 드로다운 체크
        if 'equity_curve' in portfolio:
            max_dd = self._calculate_max_drawdown(portfolio['equity_curve'])
            checks['max_drawdown'] = max_dd <= 0.20  # 20% 한계

        return checks

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """최대 드로다운 계산"""
        if len(equity_curve) == 0:
            return 0

        cumulative = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative) / cumulative
        return abs(np.min(drawdown))


class AdvancedRiskManager:
    """
    통합 고급 리스크 관리자
    """

    def __init__(self):
        self.portfolio_risk = PortfolioRiskManager()
        self.kelly = KellyCriterion()
        self.var = ValueAtRisk()
        self.stops = DynamicStopLoss()

    def analyze_trade(self,
                     symbol: str,
                     entry_price: float,
                     confidence: float,
                     market_data: pd.DataFrame,
                     portfolio_value: float,
                     current_positions: Dict = None) -> RiskMetrics:
        """
        거래 리스크 종합 분석

        Args:
            symbol: 종목 심볼
            entry_price: 진입 예정 가격
            confidence: 신호 신뢰도
            market_data: 시장 데이터
            portfolio_value: 포트폴리오 가치
            current_positions: 현재 포지션

        Returns:
            종합 리스크 메트릭
        """
        if current_positions is None:
            current_positions = {}

        # 수익률 계산
        returns = market_data['Close'].pct_change().dropna().values

        # ATR 계산
        high = market_data['High'].values
        low = market_data['Low'].values
        close = market_data['Close'].values
        atr = self._calculate_atr(high, low, close)

        # 변동성
        volatility = np.std(returns) * np.sqrt(252)

        # 추세 강도 (간단한 버전)
        sma_20 = market_data['Close'].rolling(20).mean()
        sma_50 = market_data['Close'].rolling(50).mean()
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if sma_50.iloc[-1] > 0 else 0

        # 포지션 크기 계산
        position_size = self.portfolio_risk.calculate_position_size(
            symbol, confidence, returns, portfolio_value, current_positions
        )

        # 손절/익절 계산
        stops_dict = self.stops.calculate_stops(
            entry_price, atr, volatility, trend_strength
        )

        # VaR/CVaR 계산
        var_95 = self.var.calculate_var(returns)
        cvar_95 = self.var.calculate_cvar(returns)

        # Sharpe Ratio
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Kelly Fraction
        kelly_fraction = self.kelly.calculate_from_history(returns)

        return RiskMetrics(
            position_size=position_size,
            stop_loss=stops_dict['stop_loss'],
            take_profit=stops_dict['take_profit'],
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=self.portfolio_risk._calculate_max_drawdown(close),
            sharpe_ratio=sharpe_ratio,
            kelly_fraction=kelly_fraction
        )

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Average True Range 계산"""
        if len(high) < period:
            return 0

        tr = np.maximum(
            high[-period:] - low[-period:],
            np.abs(high[-period:] - np.roll(close[-period:], 1)[1:]),
            np.abs(low[-period:] - np.roll(close[-period:], 1)[1:])
        )

        return np.mean(tr)


def main():
    """테스트 함수"""

    # 샘플 데이터 생성
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 일일 수익률
    portfolio_value = 100000

    # 리스크 관리자 생성
    risk_manager = AdvancedRiskManager()

    # 샘플 시장 데이터
    dates = pd.date_range('2023-01-01', periods=252)
    prices = 100 * np.exp(np.cumsum(returns))
    market_data = pd.DataFrame({
        'Date': dates,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    })

    # 리스크 분석
    metrics = risk_manager.analyze_trade(
        symbol='AAPL',
        entry_price=150.0,
        confidence=0.8,
        market_data=market_data,
        portfolio_value=portfolio_value,
        current_positions={}
    )

    print("\n=== Risk Analysis Results ===")
    print(f"Position Size: ${metrics.position_size:,.2f}")
    print(f"Stop Loss: ${metrics.stop_loss:.2f}")
    print(f"Take Profit: ${metrics.take_profit:.2f}")
    print(f"VaR (95%): {metrics.var_95:.4f}")
    print(f"CVaR (95%): {metrics.cvar_95:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Kelly Fraction: {metrics.kelly_fraction:.2%}")


if __name__ == "__main__":
    main()