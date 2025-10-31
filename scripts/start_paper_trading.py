"""
Paper Trading 시작 스크립트
간단한 설정으로 Paper Trading 실행
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from integrated_trading_system import IntegratedTradingSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Paper Trading 실행"""
    print("\n" + "="*70)
    print("🚀 InvenstX Paper Trading System")
    print("="*70)

    # 설정
    initial_balance = 100000
    trading_days = 7
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

    print(f"\n📋 Configuration:")
    print(f"  Initial Balance: ${initial_balance:,.2f}")
    print(f"  Trading Period: {trading_days} days")
    print(f"  Symbols: {', '.join(symbols)}")

    # 사용자 확인
    response = input("\n시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("❌ 취소되었습니다.")
        return

    # 시스템 생성
    print("\n⚙️  Initializing trading system...")
    system = IntegratedTradingSystem(
        initial_balance=initial_balance,
        symbols=symbols
    )

    # Paper Trading 실행
    print("\n🎯 Starting Paper Trading simulation...")
    print("-" * 70)

    try:
        await system.run_paper_trading(days=trading_days)

        print("\n" + "="*70)
        print("✅ Paper Trading completed successfully!")
        print("="*70)
        print(f"\n📁 Results saved in: results/paper_trading/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Paper Trading interrupted by user")
        system.save_results()
        print("📁 Partial results saved")

    except Exception as e:
        logger.error(f"Error during paper trading: {e}", exc_info=True)
        print(f"\n❌ Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())