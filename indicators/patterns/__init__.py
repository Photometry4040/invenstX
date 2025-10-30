"""
주식 차트 패턴 인식 모듈

이 모듈은 다양한 기술적 분석 패턴을 감지하고 분석하는 기능을 제공합니다.
"""

from .bollinger_bands import BollingerBands
from .double_bottom import DoubleBottom
from .pattern_analyzer import PatternAnalyzer

# 선택적 임포트 (파일이 없을 수 있음)
try:
    from .double_top import DoubleTop
except ImportError:
    pass

try:
    from .head_and_shoulders import HeadAndShoulders
except ImportError:
    pass

__all__ = [
    'BollingerBands',
    'DoubleBottom',
    'PatternAnalyzer'
]

# 선택적 모듈이 있는 경우 __all__에 추가
try:
    DoubleTop
    __all__.append('DoubleTop')
except NameError:
    pass

try:
    HeadAndShoulders
    __all__.append('HeadAndShoulders')
except NameError:
    pass

from indicators.patterns.candlestick_patterns import scan_patterns

__all__.append('scan_patterns') 