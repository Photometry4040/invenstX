# 📋 InvenstX 개발 계획서 (Development Plan)

## 🎯 프로젝트 목표
AI 기반 주식 트레이딩 시스템을 **무료**로 개인 및 외부 사용자가 활용할 수 있도록 배포

## 📅 개발 로드맵

### Phase 1: 핵심 개선 (2024년 11월 - 1주차)
- [x] 23D Feature Engineering 구현
- [x] Optuna 하이퍼파라미터 최적화
- [x] Enhanced Model Training Pipeline
- [x] 성능 비교 분석 시스템
- [x] GitHub 저장소 구성 및 CI/CD

### Phase 2: 즉시 구현 가능한 개선 (2024년 11월 - 2주차)
- [ ] **앙상블 트레이딩 시스템** ⭐ 우선순위 1
  - [x] ensemble_trading.py 구현
  - [ ] 다양한 투표 방식 테스트
  - [ ] 성능 검증 및 문서화

- [ ] **Paper Trading Mode** ⭐ 우선순위 2
  - [ ] 가상 잔고 시스템
  - [ ] 실시간 시장 데이터 연동
  - [ ] 거래 시뮬레이션 엔진

- [ ] **Kelly Criterion 포지션 사이징** ⭐ 우선순위 3
  - [ ] 수학적 최적 베팅 크기 계산
  - [ ] 리스크 관리 모듈 통합
  - [ ] 백테스팅 적용

### Phase 3: 중기 개선 (2024년 11월 - 3-4주차)
- [ ] **실시간 대시보드 개선**
  - [ ] Plotly Dash 인터랙티브 UI
  - [ ] WebSocket 실시간 업데이트
  - [ ] 모바일 반응형 디자인

- [ ] **시장 상황 인식 (Market Regime)**
  - [ ] Hidden Markov Model 구현
  - [ ] Bull/Bear/Sideways 분류
  - [ ] 전략 자동 전환

### Phase 4: 장기 개선 (2024년 12월)
- [ ] **Transformer 모델 적용**
- [ ] **뉴스 감성 분석**
- [ ] **Alternative Data 통합**
- [ ] **GPU 가속 (RAPIDS)**

## 🚀 배포 전략 (무료)

### 1. 로컬 개인 사용 (즉시 가능)
```bash
# 설치
git clone https://github.com/Photometry4040/invenstX.git
cd invenstX
pip install -r requirements.txt

# 실행
streamlit run main.py
```

### 2. 클라우드 배포 옵션 (무료 티어)

#### 2.1 **Streamlit Cloud** (추천) ⭐
**장점**:
- 완전 무료
- GitHub 연동 자동 배포
- HTTPS 제공
- 커스텀 도메인 가능

**제한사항**:
- 1GB 메모리
- 1GB 스토리지

**배포 방법**:
```yaml
# .streamlit/config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

**URL**: `https://invenstx.streamlit.app`

#### 2.2 **Hugging Face Spaces** ⭐
**장점**:
- 무료 GPU (시간 제한)
- 16GB 메모리
- 50GB 스토리지
- 커뮤니티 지원

**배포 파일** (`app.py`):
```python
import gradio as gr
from main import run_analysis

iface = gr.Interface(
    fn=run_analysis,
    inputs=[
        gr.Textbox(label="Ticker"),
        gr.Date(label="Start Date"),
        gr.Date(label="End Date")
    ],
    outputs=[
        gr.Plot(label="Price Chart"),
        gr.Dataframe(label="Performance")
    ]
)

iface.launch()
```

**URL**: `https://huggingface.co/spaces/[username]/invenstx`

#### 2.3 **Google Colab** (노트북 형태)
**장점**:
- 무료 GPU (T4)
- 12GB RAM
- Google Drive 연동

**Colab 노트북**:
```python
# InvenstX_Colab.ipynb
!git clone https://github.com/Photometry4040/invenstX.git
%cd invenstx
!pip install -r requirements.txt

# Streamlit을 ngrok으로 터널링
!pip install pyngrok
from pyngrok import ngrok

# Streamlit 앱 실행
!streamlit run main.py &

# 외부 접속 URL 생성
public_url = ngrok.connect(8501)
print(f"외부 접속 URL: {public_url}")
```

#### 2.4 **GitHub Codespaces** (월 60시간 무료)
**장점**:
- VS Code 환경
- 4코어, 8GB RAM
- 개발과 실행 동시

**설정** (`.devcontainer/devcontainer.json`):
```json
{
  "name": "InvenstX",
  "image": "mcr.microsoft.com/devcontainers/python:3.9",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {}
  },
  "postCreateCommand": "pip install -r requirements.txt",
  "forwardPorts": [8501],
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python"]
    }
  }
}
```

#### 2.5 **Render.com** (무료 티어)
**장점**:
- 자동 배포
- 커스텀 도메인
- 환경 변수 지원

**render.yaml**:
```yaml
services:
  - type: web
    name: invenstx
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run main.py --server.port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.9
```

#### 2.6 **Railway.app** (월 $5 크레딧 무료)
**장점**:
- 간편한 배포
- 자동 HTTPS
- 환경 변수 관리

**railway.json**:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run main.py --server.port $PORT"
  }
}
```

### 3. 도커 컨테이너 배포 (로컬/클라우드)

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  invenstx:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

**실행**:
```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 4. PWA (Progressive Web App) 변환

**manifest.json**:
```json
{
  "name": "InvenstX Trading System",
  "short_name": "InvenstX",
  "description": "AI-Powered Stock Trading System",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#FF6B6B",
  "background_color": "#0E1117",
  "icons": [
    {
      "src": "/static/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### 5. API 서버 배포 (FastAPI)

**api_server.py**:
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="InvenstX API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str
    date: str
    features: list

class TradingSignal(BaseModel):
    ticker: str
    action: str  # BUY, HOLD, SELL
    confidence: float
    timestamp: str

@app.get("/")
def root():
    return {"message": "InvenstX API v1.0"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # 모델 예측 로직
    signal = get_trading_signal(request.ticker, request.features)
    return signal

@app.get("/backtest/{ticker}")
async def backtest(ticker: str, start_date: str, end_date: str):
    # 백테스트 실행
    results = run_backtest(ticker, start_date, end_date)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📱 접근 방법별 비교

| 플랫폼 | 비용 | GPU | 메모리 | 스토리지 | 도메인 | 난이도 |
|-------|------|-----|--------|----------|--------|--------|
| **Streamlit Cloud** | 무료 | ❌ | 1GB | 1GB | ✅ | ⭐ |
| **Hugging Face** | 무료 | ✅ (제한) | 16GB | 50GB | ✅ | ⭐⭐ |
| **Google Colab** | 무료 | ✅ | 12GB | 15GB | ❌ | ⭐⭐ |
| **GitHub Codespaces** | 60시간/월 | ❌ | 8GB | 32GB | ❌ | ⭐⭐ |
| **Render** | 무료 | ❌ | 512MB | 10GB | ✅ | ⭐⭐ |
| **Railway** | $5 크레딧 | ❌ | 512MB | 1GB | ✅ | ⭐ |
| **로컬 Docker** | 무료 | 로컬 의존 | 로컬 의존 | 무제한 | ❌ | ⭐⭐⭐ |

## 🔐 보안 및 API 키 관리

### 환경 변수 설정
**.env.example**:
```bash
# API Keys (Optional)
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Database (Optional)
DATABASE_URL=sqlite:///invenstx.db

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

### Streamlit Secrets
**.streamlit/secrets.toml**:
```toml
[api_keys]
alpha_vantage = "your_key_here"
finnhub = "your_key_here"

[database]
url = "sqlite:///invenstx.db"
```

## 📊 성능 모니터링

### 1. 무료 모니터링 도구
- **Sentry** (에러 추적): 월 5,000 이벤트 무료
- **New Relic** (APM): 월 100GB 무료
- **Datadog** (로그): 일 500MB 무료

### 2. 자체 모니터링
```python
# monitoring.py
import psutil
import time
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def log_metrics(self):
        self.metrics.append({
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        })
```

## 🤝 커뮤니티 구축

### 1. Discord 서버
- 무료 커뮤니티 플랫폼
- 실시간 토론
- 봇 통합 가능

### 2. GitHub Discussions
- 코드 저장소와 통합
- Q&A 포럼
- 아이디어 공유

### 3. Documentation Site
- **GitHub Pages** (무료)
- **GitBook** (무료 티어)
- **Docusaurus** (무료)

## 📈 성장 전략

### Phase 1: MVP (현재)
- 핵심 기능 완성
- GitHub 공개
- 기본 문서화

### Phase 2: 커뮤니티 (1개월)
- Streamlit Cloud 배포
- Discord 서버 개설
- 초기 사용자 피드백

### Phase 3: 확장 (3개월)
- Hugging Face Spaces 배포
- API 서버 구축
- 모바일 앱 개발

### Phase 4: 수익화 옵션 (6개월)
- **GitHub Sponsors** (기부)
- **Buy Me a Coffee** (후원)
- **Premium Features** (선택적)
  - 더 많은 모델
  - 실시간 알림
  - 우선 지원

## 🎯 즉시 실행 계획

### 이번 주 목표
1. **월요일**: 앙상블 시스템 테스트
2. **화요일**: Streamlit Cloud 배포
3. **수요일**: Paper Trading 구현
4. **목요일**: Kelly Criterion 적용
5. **금요일**: 문서 업데이트 및 공개

### 배포 체크리스트
- [ ] README 업데이트
- [ ] 라이선스 확인 (MIT)
- [ ] 환경 변수 분리
- [ ] Docker 이미지 생성
- [ ] CI/CD 파이프라인 설정
- [ ] 배포 스크립트 작성
- [ ] 모니터링 설정
- [ ] 백업 전략 수립

## 💡 핵심 차별화 요소

1. **완전 무료**: 오픈소스 + 무료 배포
2. **쉬운 설치**: 원클릭 배포
3. **한국어 지원**: 한국 사용자 친화적
4. **AI 최적화**: Optuna + Ensemble
5. **실시간 지원**: WebSocket 통합
6. **클라우드 네이티브**: 어디서나 접속

## 📞 지원 및 연락처

- **GitHub**: https://github.com/Photometry4040/invenstX
- **Issues**: https://github.com/Photometry4040/invenstX/issues
- **Discussions**: https://github.com/Photometry4040/invenstX/discussions
- **Email**: invenstx@example.com (추후 설정)

## 🏆 성공 지표

### 단기 (1개월)
- [ ] GitHub Stars: 100+
- [ ] 활성 사용자: 50+
- [ ] 일일 실행: 100+

### 중기 (3개월)
- [ ] GitHub Stars: 500+
- [ ] 활성 사용자: 500+
- [ ] 커뮤니티 멤버: 100+

### 장기 (6개월)
- [ ] GitHub Stars: 1000+
- [ ] 활성 사용자: 2000+
- [ ] 기여자: 10+

---

*Last Updated: 2024-10-31*
*Version: 2.0.0*