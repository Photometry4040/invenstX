# reinforcement_learning/utils.py
"""
강화학습용 유틸리티 함수 모듈
"""
def normalize_data(data):
    """
    데이터 정규화
    
    :param data: 주식 데이터
    :return: 정규화된 데이터
    """
    return (data - data.min()) / (data.max() - data.min()) if data.max() != data.min() else data