# 🐳 vTTS Docker Guide

vTTS의 각 TTS 엔진을 Docker로 격리하여 의존성 충돌 없이 실행하는 방법입니다.

## 🎯 왜 Docker인가?

| 설치 방식 | 장점 | 단점 |
|-----------|------|------|
| `pip install vtts[all]` | 간편함 | 의존성 충돌 가능 |
| `pip install vtts[supertonic]` | 가벼움 | 하나의 엔진만 사용 |
| **Docker** | 완전한 격리, 동시 실행 | Docker 필요 |

**권장 사항:**
- 단일 엔진만 사용 → pip 설치
- 여러 엔진 동시 사용 → **Docker 권장**

---

## 📦 빠른 시작

### 1. 단일 엔진 실행

```bash
# Supertonic (가장 빠름, 다국어)
docker-compose up -d supertonic
# → http://localhost:8001

# GPT-SoVITS (음성 클로닝) - reference audio 필수!
mkdir -p reference_audio  # 참조 오디오 디렉토리 생성
docker-compose up -d gptsovits
# → http://localhost:8002

# CosyVoice (고품질)
docker-compose up -d cosyvoice
# → http://localhost:8003
```

### 2. 전체 실행 (모든 엔진)

```bash
# 모든 엔진 + API Gateway
docker-compose --profile gateway up -d
# → http://localhost:8000 (Gateway)
# → http://localhost:8001 (Supertonic)
# → http://localhost:8002 (GPT-SoVITS)
# → http://localhost:8003 (CosyVoice)
```

---

## 🔧 이미지 빌드

### 개별 빌드

```bash
# Supertonic (가장 빠름, ~5분)
docker build -f docker/Dockerfile.supertonic -t vtts:supertonic .

# GPT-SoVITS (가장 오래 걸림, ~15분)
docker build -f docker/Dockerfile.gptsovits -t vtts:gptsovits .

# CosyVoice (~10분)
docker build -f docker/Dockerfile.cosyvoice -t vtts:cosyvoice .
```

### 전체 빌드

```bash
docker-compose build
```

---

## 🌐 포트 구성

| 엔진 | 포트 | 설명 |
|------|------|------|
| Gateway (Nginx) | 8000 | API 라우팅 (선택적) |
| Supertonic | 8001 | ONNX 기반, 가장 빠름 |
| GPT-SoVITS | 8002 | 음성 클로닝 |
| CosyVoice | 8003 | ModelScope 기반 |

---

## 🚀 사용법

### Python 클라이언트

```python
from vtts.client import VTTSClient

# 개별 엔진 직접 접근
supertonic = VTTSClient("http://localhost:8001")
gptsovits = VTTSClient("http://localhost:8002")
cosyvoice = VTTSClient("http://localhost:8003")

# Gateway 통해 접근
gateway = VTTSClient("http://localhost:8000")

# TTS 요청
audio = supertonic.tts(
    text="안녕하세요, Supertonic입니다.",
    voice="F1",
    language="ko"
)
audio.save("output.wav")
```

### cURL

```bash
# Supertonic (참조 오디오 불필요)
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "F1"}' \
  --output hello.mp3

# GPT-SoVITS (참조 오디오 필수!)
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요, 음성 클로닝 테스트입니다.",
    "voice": "reference",
    "language": "ko",
    "reference_audio": "/app/reference_audio/sample.wav",
    "reference_text": "이것은 참조 오디오의 텍스트입니다."
  }' \
  --output cloned.wav
```

### GPT-SoVITS 사용법 (Python)

```python
from vtts.client import VTTSClient

# GPT-SoVITS 클라이언트
gptsovits = VTTSClient("http://localhost:8002")

# 음성 클로닝 TTS (reference_audio 필수!)
audio = gptsovits.tts(
    text="안녕하세요, 음성 클로닝 테스트입니다.",
    model="lj1995/GPT-SoVITS",
    language="ko",
    reference_audio="/app/reference_audio/sample.wav",
    reference_text="참조 오디오에서 말하는 내용"
)
audio.save("cloned.wav")
```

> ⚠️ **중요**: GPT-SoVITS는 `reference_audio`와 `reference_text`가 필수입니다!

---

## 📊 리소스 요구사항

| 엔진 | GPU 메모리 | RAM | 디스크 |
|------|-----------|-----|--------|
| Supertonic | ~1GB | 4GB | 500MB |
| GPT-SoVITS | ~4GB | 8GB | 5GB |
| CosyVoice | ~3GB | 8GB | 3GB |
| **전체** | **~8GB** | **16GB** | **10GB** |

---

## ⚙️ 고급 설정

### GPU 할당

여러 GPU가 있는 경우, 각 엔진에 다른 GPU를 할당:

```yaml
# docker-compose.override.yml
services:
  supertonic:
    environment:
      - CUDA_VISIBLE_DEVICES=0

  gptsovits:
    environment:
      - CUDA_VISIBLE_DEVICES=1

  cosyvoice:
    environment:
      - CUDA_VISIBLE_DEVICES=2
```

### 모델 캐시 공유

기본적으로 HuggingFace 캐시는 Docker 볼륨으로 공유됩니다:

```bash
# 캐시 확인
docker volume ls | grep vtts

# 캐시 삭제 (모델 재다운로드 필요)
docker volume rm vtts-hf-cache
```

### CPU 전용 모드

GPU 없이 실행:

```bash
# docker-compose.override.yml 생성
cat > docker-compose.override.yml << 'EOF'
services:
  supertonic:
    deploy:
      resources:
        reservations:
          devices: []
    command: ["Supertone/supertonic-2", "--device", "cpu"]
EOF

docker-compose up -d supertonic
```

---

## 🔍 로그 및 디버깅

```bash
# 실시간 로그
docker-compose logs -f supertonic

# 컨테이너 접속
docker exec -it vtts-supertonic bash

# 헬스체크
curl http://localhost:8001/health
```

---

## 🛑 종료 및 정리

```bash
# 서비스 종료
docker-compose down

# 이미지 삭제
docker-compose down --rmi all

# 볼륨까지 삭제 (모델 캐시 포함)
docker-compose down -v
```

---

## 🆚 엔진 비교

| 특성 | Supertonic | GPT-SoVITS v3 | CosyVoice |
|------|------------|---------------|-----------|
| 속도 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 품질 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 다국어 | ✅ 5개 언어 | ✅ 5개 언어 | ✅ 다국어 |
| 음성 클로닝 | ❌ | ✅ Zero-shot | ⚠️ 제한적 |
| 메모리 | 가벼움 | 무거움 | 중간 |
| 참조 오디오 | 불필요 | **필수** | 선택적 |
| 설치 난이도 | 쉬움 | 어려움 | 중간 |

> **GPT-SoVITS 참조 오디오**: GPT-SoVITS는 zero-shot 음성 클로닝 모델이므로 합성할 음성의 참조 오디오가 반드시 필요합니다.

---

## 🆘 문제 해결

### GPU를 찾을 수 없음

```bash
# NVIDIA Docker 런타임 확인
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 포트 충돌

```bash
# 사용 중인 포트 확인
netstat -tlnp | grep 800

# 다른 포트로 변경
docker-compose up -d -e "8011:8000" supertonic
```

### 메모리 부족

```yaml
# docker-compose.override.yml
services:
  gptsovits:
    deploy:
      resources:
        limits:
          memory: 12G
```

---

## 📁 파일 구조

```
vTTS/
├── docker/
│   ├── Dockerfile.supertonic    # Supertonic 이미지
│   ├── Dockerfile.gptsovits     # GPT-SoVITS 이미지
│   ├── Dockerfile.cosyvoice     # CosyVoice 이미지
│   └── nginx.conf               # API Gateway 설정
├── docker-compose.yml           # 오케스트레이션
└── DOCKER.md                    # 이 문서
```
