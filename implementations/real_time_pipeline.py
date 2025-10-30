"""
실시간 데이터 파이프라인
WebSocket을 통한 실시간 주가 데이터 스트리밍 및 처리
"""

import asyncio
import json
import logging
from datetime import datetime
from collections import deque
from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import websocket
import redis
import yfinance as yf

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """실시간 틱 데이터"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int


class RealTimeDataPipeline:
    """
    실시간 데이터 파이프라인
    - WebSocket 연결 관리
    - 데이터 버퍼링 및 캐싱
    - 실시간 특징 계산
    """

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 buffer_size: int = 10000):

        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.data_buffers = {}  # 심볼별 데이터 버퍼
        self.buffer_size = buffer_size
        self.subscribers = []  # 데이터 구독자 목록
        self.websocket_connections = {}
        self.running = False

        # 기술적 지표 계산기
        self.indicator_window = 20
        self.feature_calculators = {}

    def subscribe(self, callback: Callable):
        """데이터 구독자 추가"""
        self.subscribers.append(callback)
        logger.info(f"Added subscriber: {callback.__name__}")

    async def connect_yahoo_finance(self, symbols: List[str]):
        """Yahoo Finance WebSocket 연결 (예시)"""
        # 실제로는 Yahoo Finance는 공식 WebSocket API를 제공하지 않음
        # 대신 yfinance를 폴링하거나 다른 데이터 소스 사용
        logger.info(f"Connecting to data source for symbols: {symbols}")

        for symbol in symbols:
            self.data_buffers[symbol] = deque(maxlen=self.buffer_size)
            asyncio.create_task(self._poll_yahoo_data(symbol))

    async def _poll_yahoo_data(self, symbol: str, interval: float = 1.0):
        """Yahoo Finance 데이터 폴링 (실시간 시뮬레이션)"""
        while self.running:
            try:
                # 실시간 데이터 가져오기
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # 현재 가격 정보
                current_price = info.get('regularMarketPrice', 0)
                volume = info.get('regularMarketVolume', 0)
                bid = info.get('bid', current_price)
                ask = info.get('ask', current_price)
                bid_size = info.get('bidSize', 100)
                ask_size = info.get('askSize', 100)

                # TickData 생성
                tick = TickData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    volume=volume,
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size
                )

                # 데이터 처리
                await self.process_tick(tick)

                # 대기
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error polling data for {symbol}: {e}")
                await asyncio.sleep(5)  # 에러 시 더 긴 대기

    async def connect_alpaca_websocket(self, symbols: List[str], api_key: str, api_secret: str):
        """Alpaca WebSocket 연결 (실제 실시간)"""
        import alpaca_trade_api as tradeapi

        # Alpaca 연결 설정
        conn = tradeapi.stream.Stream(
            api_key,
            api_secret,
            base_url='wss://stream.data.alpaca.markets/v2',
            data_feed='iex'  # or 'sip' for premium
        )

        async def on_trade(t):
            """거래 데이터 수신"""
            tick = TickData(
                symbol=t.symbol,
                timestamp=t.timestamp,
                price=t.price,
                volume=t.size,
                bid=t.price,  # 실제로는 quote에서 가져와야 함
                ask=t.price,
                bid_size=0,
                ask_size=0
            )
            await self.process_tick(tick)

        # 심볼 구독
        for symbol in symbols:
            conn.subscribe_trades(on_trade, symbol)
            self.data_buffers[symbol] = deque(maxlen=self.buffer_size)

        # WebSocket 시작
        await conn.run()

    async def process_tick(self, tick: TickData):
        """틱 데이터 처리"""
        try:
            # 버퍼에 추가
            self.data_buffers[tick.symbol].append(tick)

            # Redis 캐싱
            self._cache_tick(tick)

            # 기술적 지표 계산
            features = await self.calculate_features(tick.symbol)

            # 구독자에게 알림
            await self._notify_subscribers(tick, features)

        except Exception as e:
            logger.error(f"Error processing tick: {e}")

    def _cache_tick(self, tick: TickData):
        """Redis에 틱 데이터 캐싱"""
        key = f"tick:{tick.symbol}:{tick.timestamp.timestamp()}"
        value = {
            'price': tick.price,
            'volume': tick.volume,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid
        }

        # Redis에 저장 (TTL: 1시간)
        self.redis_client.setex(
            key,
            3600,
            json.dumps(value)
        )

        # 최신 가격 업데이트
        self.redis_client.set(f"latest:{tick.symbol}", tick.price)

    async def calculate_features(self, symbol: str) -> Dict:
        """실시간 특징 계산"""
        if symbol not in self.data_buffers or len(self.data_buffers[symbol]) < 2:
            return {}

        # 최근 데이터 추출
        recent_data = list(self.data_buffers[symbol])
        prices = [tick.price for tick in recent_data]
        volumes = [tick.volume for tick in recent_data]

        features = {}

        # 기본 통계
        if len(prices) >= 2:
            features['return_1m'] = (prices[-1] - prices[-2]) / prices[-2]
            features['volume_ratio'] = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1

        # 이동 평균
        if len(prices) >= 5:
            features['sma_5'] = np.mean(prices[-5:])
            features['price_to_sma5'] = prices[-1] / features['sma_5']

        if len(prices) >= 20:
            features['sma_20'] = np.mean(prices[-20:])
            features['price_to_sma20'] = prices[-1] / features['sma_20']

            # 볼린저 밴드
            std_20 = np.std(prices[-20:])
            features['bb_upper'] = features['sma_20'] + 2 * std_20
            features['bb_lower'] = features['sma_20'] - 2 * std_20
            features['bb_position'] = (prices[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # RSI (간단 버전)
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, 14):
                change = prices[-i] - prices[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100 if avg_gain > 0 else 50

        # 스프레드 분석
        if recent_data[-1].bid > 0 and recent_data[-1].ask > 0:
            features['spread'] = recent_data[-1].ask - recent_data[-1].bid
            features['spread_pct'] = features['spread'] / recent_data[-1].price

        # 모멘텀
        if len(prices) >= 10:
            features['momentum_10'] = (prices[-1] - prices[-10]) / prices[-10]

        # 변동성
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            features['volatility'] = np.std(returns) * np.sqrt(252)  # 연율화

        return features

    async def _notify_subscribers(self, tick: TickData, features: Dict):
        """구독자에게 데이터 전달"""
        data = {
            'tick': tick,
            'features': features,
            'timestamp': datetime.now()
        }

        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(data)
                else:
                    subscriber(data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    async def get_recent_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """최근 데이터 반환"""
        if symbol not in self.data_buffers:
            return pd.DataFrame()

        recent_ticks = list(self.data_buffers[symbol])[-limit:]

        if not recent_ticks:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid
            }
            for tick in recent_ticks
        ])

        df.set_index('timestamp', inplace=True)
        return df

    async def start(self, symbols: List[str]):
        """파이프라인 시작"""
        logger.info(f"Starting real-time pipeline for {symbols}")
        self.running = True

        # Yahoo Finance 연결 (또는 다른 데이터 소스)
        await self.connect_yahoo_finance(symbols)

        # 메인 루프
        while self.running:
            await asyncio.sleep(1)

            # 상태 체크 및 로깅
            for symbol in symbols:
                buffer_size = len(self.data_buffers.get(symbol, []))
                logger.debug(f"{symbol}: {buffer_size} ticks in buffer")

    async def stop(self):
        """파이프라인 중지"""
        logger.info("Stopping real-time pipeline")
        self.running = False

        # WebSocket 연결 종료
        for conn in self.websocket_connections.values():
            if hasattr(conn, 'close'):
                conn.close()

        # Redis 연결 종료
        self.redis_client.close()


class StreamingFeatureEngine:
    """스트리밍 특징 엔진"""

    def __init__(self, pipeline: RealTimeDataPipeline):
        self.pipeline = pipeline
        self.feature_cache = {}

        # 파이프라인 구독
        pipeline.subscribe(self.on_new_data)

    async def on_new_data(self, data: Dict):
        """새 데이터 수신 시 특징 업데이트"""
        tick = data['tick']
        features = data['features']

        # 추가 특징 계산
        enhanced_features = await self.calculate_advanced_features(tick, features)

        # 캐시 업데이트
        self.feature_cache[tick.symbol] = enhanced_features

        logger.debug(f"Updated features for {tick.symbol}: {len(enhanced_features)} features")

    async def calculate_advanced_features(self, tick: TickData, basic_features: Dict) -> Dict:
        """고급 특징 계산"""
        features = basic_features.copy()

        # 주문 불균형
        if tick.bid_size > 0 or tick.ask_size > 0:
            features['order_imbalance'] = (tick.bid_size - tick.ask_size) / (tick.bid_size + tick.ask_size)

        # 틱 방향
        symbol = tick.symbol
        if symbol in self.pipeline.data_buffers and len(self.pipeline.data_buffers[symbol]) >= 2:
            prev_tick = list(self.pipeline.data_buffers[symbol])[-2]
            features['tick_direction'] = 1 if tick.price > prev_tick.price else -1 if tick.price < prev_tick.price else 0

        # VWAP (Volume Weighted Average Price)
        df = await self.pipeline.get_recent_data(symbol, 20)
        if not df.empty and 'volume' in df.columns:
            vwap = (df['price'] * df['volume']).sum() / df['volume'].sum()
            features['vwap'] = vwap
            features['price_to_vwap'] = tick.price / vwap if vwap > 0 else 1

        return features


async def main():
    """메인 테스트 함수"""

    # 파이프라인 생성
    pipeline = RealTimeDataPipeline()

    # 특징 엔진 생성
    feature_engine = StreamingFeatureEngine(pipeline)

    # 데이터 처리 콜백
    async def on_data(data):
        tick = data['tick']
        features = data['features']
        print(f"\n[{tick.timestamp}] {tick.symbol}")
        print(f"  Price: ${tick.price:.2f}")
        print(f"  Volume: {tick.volume:,}")
        if features:
            print(f"  RSI: {features.get('rsi', 'N/A'):.2f}")
            print(f"  Volatility: {features.get('volatility', 'N/A'):.2%}")

    # 구독
    pipeline.subscribe(on_data)

    # 시작
    symbols = ['AAPL', 'TSLA', 'GOOGL']

    try:
        await pipeline.start(symbols)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        await pipeline.stop()


if __name__ == "__main__":
    # Redis 서버가 실행 중이어야 함
    # brew install redis && redis-server (macOS)
    # 또는 Docker: docker run -d -p 6379:6379 redis:alpine

    asyncio.run(main())