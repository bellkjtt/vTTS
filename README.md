# vTTS - Universal TTS/STT Serving System

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**vLLM for Speech** - Huggingface에서 바로 다운로드하여 추론 가능한 범용 TTS/STT 서빙 시스템

한국어 | [English](README_EN.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

## 🎯 목표

- 🚀 **간단한 사용법**: `vtts serve model-name` 한 줄로 서버 실행
- 🤗 **Huggingface 통합**: 모델 자동 다운로드 및 캐싱
- 🌐 **OpenAI 호환 API**: OpenAI TTS & Whisper API와 완전 호환
- 🎙️ **TTS + STT 통합**: 텍스트 음성 변환과 음성 인식 동시 지원
- 🇰🇷 **한국어 우선**: 한국어 지원 모델 중심
- 🐳 **Docker 지원**: 의존성 충돌 없이 여러 엔진 동시 실행
- 🎮 **CUDA 지원**: GPU 가속으로 빠른 추론

## 📦 지원 모델

### TTS (Text-to-Speech)
| 엔진 | 속도 | 품질 | 다국어 | 음성 클로닝 | 참조 오디오 |
|------|------|------|--------|------------|------------|
| ✅ **Supertonic-2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 5개 언어 | ❌ | 불필요 |
| ✅ **GPT-SoVITS v3** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 5개 언어 | ✅ Zero-shot | **필수** |
| ✅ **CosyVoice3** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 9개 언어 | ⚠️ | 선택적 |
| 🔜 **StyleTTS2**, **XTTS-v2**, **Bark** | - | - | - | - | - |

> **GPT-SoVITS**: Zero-shot 음성 클로닝 모델로, 합성할 목표 음성의 참조 오디오(3~10초)가 필수입니다.

### STT (Speech-to-Text)
- ✅ **Faster-Whisper** - 초고속 Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

---

## 🚀 빠른 시작

### 방법 1: Supertonic만 사용 (가장 간편)

```bash
# CUDA 지원 설치 (권장)
pip install "vtts[supertonic-cuda] @ git+https://github.com/bellkjtt/vTTS.git"

# CPU만 사용
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# 서버 실행
vtts serve Supertone/supertonic-2 --device cuda
```

### 방법 2: GPT-SoVITS 설치 (음성 클로닝)

> ⚠️ GPT-SoVITS는 저장소 클론이 **필수**입니다 (pip 패키지 없음)

```bash
# 1. vTTS 기본 설치
pip install "vtts[gptsovits] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. GPT-SoVITS 저장소 클론 (필수!)
git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
cd third_party/GPT-SoVITS
pip install -r requirements.txt
cd ../..

# 3. 환경변수 설정 (선택적)
export GPT_SOVITS_PATH=$(pwd)/third_party/GPT-SoVITS

# 4. 서버 실행
vtts serve lj1995/GPT-SoVITS --device cuda --port 8002
```

### 방법 3: Docker (의존성 충돌 방지, 권장)

```bash
# Supertonic (가장 빠름)
docker-compose up -d supertonic   # :8001

# GPT-SoVITS (음성 클로닝) - reference_audio 볼륨 필요
mkdir -p reference_audio
docker-compose up -d gptsovits    # :8002

# CosyVoice (고품질)
docker-compose up -d cosyvoice    # :8003

# 전체 + API Gateway
docker-compose --profile gateway up -d  # :8000
```

📖 자세한 내용: [Docker 가이드](DOCKER.md)

### 방법 4: CLI 자동 설치

```bash
# 기본 설치 후 엔진 추가
pip install git+https://github.com/bellkjtt/vTTS.git

vtts setup --engine supertonic --cuda   # Supertonic + CUDA
vtts setup --engine gptsovits           # GPT-SoVITS (저장소 클론 포함)
vtts setup --engine all                 # 모든 엔진
```

---

## 🔧 환경 설정

### 환경 진단 및 자동 수정

```bash
# 환경 진단
vtts doctor

# 자동 수정 (numpy, onnxruntime 호환성 문제 해결)
vtts doctor --fix

# CUDA 지원 강제 설치
vtts doctor --fix --cuda
```

출력 예시:
```
🩺 vTTS Environment Diagnosis

✓ Python: 3.10.12
✓ numpy: 1.26.4
✓ onnxruntime: 1.16.0 (CUDA 지원)
  Providers: CUDAExecutionProvider, CPUExecutionProvider
✓ PyTorch: 2.1.0 (CUDA 12.1)
  GPU: NVIDIA GeForce RTX 4090
✓ vTTS: 설치됨

✅ 모든 환경이 정상입니다!
```

### Kaggle/Colab에서

```python
# 설치 + 환경 자동 설정
!pip install -q git+https://github.com/bellkjtt/vTTS.git
!vtts doctor --fix --cuda
```

---

## 💻 서버 실행

### Supertonic (빠른 TTS)
```bash
vtts serve Supertone/supertonic-2
vtts serve Supertone/supertonic-2 --device cuda --port 8000
```

### GPT-SoVITS (음성 클로닝)
```bash
# GPT-SoVITS 저장소 클론 필요!
git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
cd third_party/GPT-SoVITS && pip install -r requirements.txt && cd ../..

# 서버 실행
vtts serve lj1995/GPT-SoVITS --device cuda --port 8002
```

### TTS + STT 동시
```bash
vtts serve Supertone/supertonic-2 --stt-model large-v3
vtts serve Supertone/supertonic-2 --stt-model base --device cuda
```

### 사용 가능한 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | 0.0.0.0 | 서버 호스트 |
| `--port` | 8000 | 서버 포트 |
| `--device` | auto | cuda, cpu, auto |
| `--stt-model` | None | Whisper 모델 (base, large-v3 등) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

---

## 🐍 Python 사용

### 기본 사용법
```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")

# TTS
audio = client.tts(
    text="안녕하세요, vTTS입니다.",
    voice="F1",
    language="ko",
    speed=1.05
)
audio.save("output.wav")

# STT
text = client.stt("audio.wav")
print(text)
```

### 고급 옵션 (Supertonic)
```python
audio = client.tts(
    text="안녕하세요",
    voice="F1",           # M1-M4, F1-F4
    language="ko",        # en, ko, es, pt, fr
    speed=1.05,           # 속도 (기본: 1.05)
    total_steps=5,        # 품질 (1-20, 기본: 5)
    silence_duration=0.3  # 청크 간 무음 (초)
)
```

### 음성 클로닝 (GPT-SoVITS)
```python
from vtts import VTTSClient

# GPT-SoVITS 클라이언트 (참조 오디오 필수!)
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="안녕하세요, 음성 클로닝 테스트입니다.",
    model="lj1995/GPT-SoVITS",
    voice="reference",
    language="ko",
    reference_audio="./samples/reference.wav",  # 참조 오디오 (필수!)
    reference_text="참조 오디오에서 말하는 내용"  # 참조 텍스트 (필수!)
)
audio.save("cloned_voice.wav")
```
> ⚠️ GPT-SoVITS는 `reference_audio`와 `reference_text` 파라미터가 필수입니다!

### OpenAI SDK 호환
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="안녕하세요, 반갑습니다."
)
response.stream_to_file("output.mp3")
```

### cURL
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "F1", "model": "Supertone/supertonic-2"}' \
  --output output.mp3
```

---

## 🐳 Docker

### 포트 구성
| 엔진 | 포트 | GPU 메모리 |
|------|------|-----------|
| Gateway (Nginx) | 8000 | - |
| Supertonic | 8001 | ~1GB |
| GPT-SoVITS | 8002 | ~4GB |
| CosyVoice | 8003 | ~3GB |

### 빠른 시작
```bash
# 이미지 빌드
docker-compose build

# 실행
docker-compose up -d supertonic   # Supertonic만
docker-compose up -d              # 전체

# 로그
docker-compose logs -f supertonic

# 종료
docker-compose down
```

📖 자세한 내용: [Docker 가이드](DOCKER.md)

---

## 📊 CLI 명령어

| 명령어 | 설명 |
|--------|------|
| `vtts serve MODEL` | TTS 서버 시작 |
| `vtts doctor` | 환경 진단 |
| `vtts doctor --fix` | 환경 자동 수정 |
| `vtts setup --engine ENGINE` | 엔진별 설치 |
| `vtts list-models` | 지원 모델 목록 |
| `vtts info MODEL` | 모델 정보 |

---

## 🏗️ 아키텍처

```
vTTS/
├── vtts/
│   ├── __init__.py           # 환경 자동 체크
│   ├── cli.py                # CLI (serve, doctor, setup)
│   ├── client.py             # Python 클라이언트
│   ├── server/
│   │   ├── app.py            # FastAPI 앱
│   │   ├── routes.py         # TTS API 라우트
│   │   ├── stt_routes.py     # STT API 라우트
│   │   └── models.py         # Pydantic 모델
│   ├── engines/
│   │   ├── base.py           # 베이스 엔진 인터페이스
│   │   ├── registry.py       # 엔진 자동 등록
│   │   ├── supertonic.py     # Supertonic 엔진
│   │   ├── gptsovits.py      # GPT-SoVITS 엔진
│   │   ├── cosyvoice.py      # CosyVoice 엔진
│   │   └── _supertonic/      # 내장 ONNX 모듈
│   └── utils/
│       └── audio.py          # 오디오 처리
├── docker/
│   ├── Dockerfile.supertonic
│   ├── Dockerfile.gptsovits
│   ├── Dockerfile.cosyvoice
│   └── nginx.conf            # API Gateway
├── docker-compose.yml
├── setup.py
└── README.md
```

---

## 🔧 개발 로드맵

- [x] 프로젝트 구조 설계
- [x] 베이스 엔진 인터페이스 구현
- [x] Supertonic-2 엔진 구현
- [x] CosyVoice3 엔진 구현
- [x] GPT-SoVITS 엔진 구현
- [x] FastAPI 서버 구현
- [x] OpenAI 호환 API
- [x] CLI 구현 (serve, doctor, setup)
- [x] 모델 자동 다운로드
- [x] CUDA 지원
- [x] Docker 이미지
- [x] 환경 자동 진단/수정
- [ ] 스트리밍 지원
- [ ] 배치 추론 최적화

---

## 📚 문서

- [빠른 시작 가이드](QUICKSTART.md)
- [문제 해결 가이드](TROUBLESHOOTING.md)
- [Docker 가이드](DOCKER.md)
- [Kaggle 테스트 노트북](kaggle_test_notebook.ipynb)
- [예제 코드](examples/)

---

## ⚠️ 문제 해결

### numpy 호환성 에러
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**해결**: `vtts doctor --fix`

### CUDA를 찾을 수 없음
```
WARNING: CUDA requested but CUDAExecutionProvider not available
```
**해결**: `vtts doctor --fix --cuda`

### 의존성 충돌
**해결**: Docker 사용 권장
```bash
docker-compose up -d supertonic
```

📖 더 많은 문제: [문제 해결 가이드](TROUBLESHOOTING.md)

---

## 📝 라이선스

MIT License

## 💖 후원

이 프로젝트가 도움이 되셨나요? 

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

## 🙏 감사의 말

- [vLLM](https://github.com/vllm-project/vllm) - 아키텍처 영감
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
