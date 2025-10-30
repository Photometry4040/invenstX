# 🚀 InvenstX 배포 가이드 (Deployment Guide)

## 📌 빠른 시작 (Quick Start)

### 옵션 1: 로컬 실행 (즉시 가능)
```bash
# 1. 저장소 클론
git clone https://github.com/Photometry4040/invenstX.git
cd invenstX

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 앱 실행
streamlit run main.py
```

브라우저에서 http://localhost:8501 접속

### 옵션 2: Docker 실행
```bash
# 1. Docker 이미지 빌드
docker build -t invenstx .

# 2. 컨테이너 실행
docker run -p 8501:8501 invenstx
```

## 🌐 무료 클라우드 배포

### 1. Streamlit Cloud (추천) ⭐⭐⭐

**장점**: 완전 무료, GitHub 자동 연동, HTTPS 제공

**배포 방법**:

1. **GitHub 저장소 준비**
   ```bash
   # 필수 파일 확인
   - requirements.txt
   - .streamlit/config.toml
   - main.py
   ```

2. **Streamlit Cloud 접속**
   - https://streamlit.io/cloud 방문
   - GitHub 계정으로 로그인

3. **새 앱 배포**
   - "New app" 클릭
   - Repository: `Photometry4040/invenstX`
   - Branch: `main`
   - Main file path: `main.py`
   - App URL: `invenstx` (원하는 이름)

4. **환경 변수 설정** (선택사항)
   - Advanced settings → Secrets
   - `.streamlit/secrets.toml` 내용 복사

5. **배포 완료**
   - URL: `https://invenstx.streamlit.app`
   - 자동 HTTPS 적용
   - 코드 푸시 시 자동 재배포

### 2. Hugging Face Spaces ⭐⭐

**장점**: 무료 GPU, 큰 메모리, ML 커뮤니티

**배포 방법**:

1. **Hugging Face 계정 생성**
   - https://huggingface.co 가입

2. **새 Space 생성**
   ```
   - Space name: invenstx
   - Select SDK: Streamlit
   - Space hardware: CPU basic (무료)
   - Visibility: Public
   ```

3. **파일 업로드**
   ```bash
   # Git으로 푸시
   git remote add hf https://huggingface.co/spaces/[username]/invenstx
   git push hf main
   ```

4. **app.py 생성** (필요시)
   ```python
   # Streamlit 앱 실행
   import subprocess
   subprocess.run(["streamlit", "run", "main.py"])
   ```

5. **접속**
   - URL: `https://huggingface.co/spaces/[username]/invenstx`

### 3. Google Colab 노트북 ⭐

**장점**: 무료 GPU (T4), 12GB RAM

**Colab 노트북 생성**:

```python
# InvenstX_Colab.ipynb

# 1. 저장소 클론
!git clone https://github.com/Photometry4040/invenstX.git
%cd invenstx

# 2. 의존성 설치
!pip install -r requirements.txt

# 3. ngrok 설치 (외부 접속용)
!pip install pyngrok
from pyngrok import ngrok

# 4. Streamlit 실행
!streamlit run main.py --server.port 8501 &

# 5. 외부 접속 URL 생성
public_url = ngrok.connect(8501, "http")
print(f"🌐 외부 접속 URL: {public_url}")
print("이 URL을 통해 누구나 접속할 수 있습니다!")
```

**공유 방법**:
- 노트북 상단 → Share → Anyone with link
- 실행 후 ngrok URL 공유

### 4. Render.com 배포 ⭐⭐

**장점**: 자동 배포, 커스텀 도메인

1. **render.yaml 생성**:
```yaml
services:
  - type: web
    name: invenstx
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run main.py --server.port $PORT --server.address 0.0.0.0"
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

2. **Render Dashboard**:
   - New → Web Service
   - Connect GitHub repo
   - 자동 배포 시작

3. **URL**: `https://invenstx.onrender.com`

## 🔧 고급 설정

### API 서버 배포 (FastAPI)

**api_server.py**:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="InvenstX API")

# CORS 설정 (모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "InvenstX API v1.0", "status": "running"}

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    # 예측 로직
    return {"ticker": ticker, "action": "BUY", "confidence": 0.85}

@app.get("/backtest/{ticker}")
async def backtest(ticker: str, days: int = 30):
    # 백테스트 로직
    return {"ticker": ticker, "return": 0.15, "sharpe": 1.2}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**배포**:
```bash
# Heroku (무료 티어 종료됨)
# Railway.app 사용 추천
railway up
```

### PWA 변환 (오프라인 지원)

**static/manifest.json**:
```json
{
  "name": "InvenstX Trading System",
  "short_name": "InvenstX",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#FF6B6B",
  "background_color": "#0E1117",
  "icons": [
    {
      "src": "/static/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

**service-worker.js**:
```javascript
// 오프라인 캐싱
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('v1').then((cache) => {
      return cache.addAll([
        '/',
        '/static/style.css',
        '/static/app.js'
      ]);
    })
  );
});
```

## 📊 배포 옵션 비교표

| 플랫폼 | 비용 | 난이도 | GPU | 메모리 | 도메인 | 추천도 |
|--------|------|--------|-----|--------|--------|--------|
| **로컬** | 무료 | ⭐ | 로컬 의존 | 로컬 의존 | ❌ | ⭐⭐⭐ |
| **Streamlit Cloud** | 무료 | ⭐ | ❌ | 1GB | ✅ | ⭐⭐⭐⭐⭐ |
| **Hugging Face** | 무료 | ⭐⭐ | ✅ | 16GB | ✅ | ⭐⭐⭐⭐ |
| **Google Colab** | 무료 | ⭐⭐ | ✅ | 12GB | ❌ | ⭐⭐⭐ |
| **Render** | 무료 | ⭐⭐ | ❌ | 512MB | ✅ | ⭐⭐⭐ |
| **Docker** | 무료 | ⭐⭐⭐ | 로컬 의존 | 설정 가능 | ❌ | ⭐⭐⭐⭐ |

## 🔒 보안 설정

### 환경 변수 관리

**.env 파일** (로컬용):
```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Security
SECRET_KEY=generate_random_key
JWT_SECRET=another_random_key

# Database
DATABASE_URL=sqlite:///invenstx.db
```

**Python에서 사용**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
```

### HTTPS 설정

**로컬 HTTPS** (개발용):
```bash
# 자체 서명 인증서 생성
openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365

# Streamlit HTTPS 실행
streamlit run main.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem
```

## 🚦 모니터링

### 무료 모니터링 서비스

1. **UptimeRobot** (무료)
   - 5분마다 상태 체크
   - 다운타임 알림
   - https://uptimerobot.com

2. **Sentry** (에러 추적)
   ```python
   import sentry_sdk
   sentry_sdk.init(
       dsn="your_sentry_dsn",
       traces_sample_rate=0.1
   )
   ```

3. **Google Analytics**
   ```html
   <!-- main.py에 추가 -->
   st.components.v1.html("""
   <script async src="https://www.googletagmanager.com/gtag/js?id=GA_ID"></script>
   """)
   ```

## 📝 체크리스트

### 배포 전 확인사항
- [ ] requirements.txt 최신화
- [ ] .gitignore 확인 (secrets 제외)
- [ ] 환경 변수 분리
- [ ] 에러 처리 강화
- [ ] 로딩 상태 표시
- [ ] 모바일 반응형 확인

### 배포 후 확인사항
- [ ] 모든 기능 테스트
- [ ] 성능 모니터링
- [ ] 에러 로그 확인
- [ ] 사용자 피드백 수집
- [ ] 백업 설정
- [ ] 자동 재시작 설정

## 🆘 문제 해결

### 일반적인 문제

**1. 메모리 부족**
```python
# 메모리 최적화
import gc
gc.collect()

# 데이터 청크 처리
for chunk in pd.read_csv('data.csv', chunksize=1000):
    process(chunk)
```

**2. 포트 충돌**
```bash
# 다른 포트 사용
streamlit run main.py --server.port 8502
```

**3. 의존성 충돌**
```bash
# 클린 설치
pip install --upgrade --force-reinstall -r requirements.txt
```

## 📞 지원

- **GitHub Issues**: https://github.com/Photometry4040/invenstX/issues
- **Discord**: [커뮤니티 링크]
- **Email**: support@invenstx.com

## 🎉 성공!

배포가 완료되면:
1. URL을 친구들과 공유하세요
2. 피드백을 받아 개선하세요
3. GitHub에 스타를 눌러주세요
4. 커뮤니티에 참여하세요

**Happy Trading! 🚀**