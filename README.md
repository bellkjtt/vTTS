# vTTS - Universal TTS/STT Serving System

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**vLLM for Speech** - Huggingface에서 바로 다운로드하여 추론 가능한 범용 TTS/STT 서빙 시스템

한국어 | [English](docs/i18n/README_EN.md) | [中文](docs/i18n/README_ZH.md) | [日本語](docs/i18n/README_JA.md)

## 목표

- **간단한 사용법**: `vtts serve model-name` 한 줄로 서버 실행
- **Huggingface 통합**: 모델 자동 다운로드 및 캐싱
- **OpenAI 호환 API**: OpenAI TTS & Whisper API와 완전 호환
- **TTS + STT 통합**: 텍스트 음성 변환과 음성 인식 동시 지원
- **한국어 우선**: 한국어 지원 모델 중심
- **Docker 지원**: 의존성 충돌 없이 여러 엔진 동시 실행
- **CUDA 지원**: GPU 가속으로 빠른 추론

## 지원 모델

### TTS (Text-to-Speech)
| 엔진 | 생성 시간* | 품질 | 다국어 | 음성 클로닝 | 참조 오디오 |
|------|-----------|------|--------|------------|------------|
| **Supertonic-2** | **0.8초** | Good | 5개 언어 | No | 불필요 |
| **Qwen3-TTS** | 5~7초 | **Excellent** | 10개 언어 | Zero-shot | 선택적 |
| **GPT-SoVITS v3** | 7~8초 | Excellent | 5개 언어 | Zero-shot | **필수** |
| **CosyVoice2** | 11초 | Very Good | 9개 언어 | Zero-shot | 선택적 |

> *생성 시간: 동일 텍스트(~50자) 기준, GPU (RTX 4090)

### Qwen3-TTS 모델 종류
| 모델 | 크기 | 참조 오디오 | 특징 |
|------|------|------------|------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | 불필요 | 9개 프리셋 스피커 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 불필요 | 9개 프리셋 스피커 (고품질) |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | **필요** | Voice Clone |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | **필요** | Voice Clone (고품질) |

> **GPT-SoVITS**: Zero-shot 음성 클로닝 모델로, 참조 오디오(3~10초) + 참조 텍스트가 필수입니다.

### STT (Speech-to-Text)
- **Faster-Whisper** - 초고속 Whisper (CTranslate2)
- **Whisper.cpp**, **Parakeet** (Coming Soon)

---

## 빠른 시작

> **CRITICAL - 버전 충돌 경고!**  
> - **CosyVoice**: `transformers==4.51.3`, `torch==2.3.1` (정확히 고정)
> - **GPT-SoVITS**: `transformers<=4.50`, `torch` 제약 없음  
> - **충돌**: CosyVoice와 GPT-SoVITS는 **동일 환경에 설치 불가!**
> 
> **권장 설치 방법:**
> 1. **Supertonic 단독**: 충돌 없음, 가장 가벼움
> 2. **Supertonic + GPT-SoVITS**: 호환 보장 (추천!)
> 3. **여러 엔진 동시 사용**: Docker 필수!

### 로컬 설치 (uv 권장 - 빠르고 정확함!)

> **uv 사용을 강력히 권장합니다!**  
> `uv`는 Rust 기반의 초고속 Python 패키지 관리자로, `pip`보다 10-100배 빠르고 의존성 해결이 정확합니다.

```bash
# uv 설치 (한 번만)
pip install uv
# 또는: curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 옵션 1: Supertonic만 (가장 가볍고 빠름)

```bash
# GPU 자동 지원 (uv 사용!)
uv pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git" --system

# 서버 실행
vtts serve Supertone/supertonic-2 --device cuda
```

#### 옵션 2: Supertonic + GPT-SoVITS (호환 보장!)

```bash
# 1. 통합 설치 (uv 사용!)
uv pip install "vtts[supertonic-gptsovits] @ git+https://github.com/bellkjtt/vTTS.git" --system

# 2. GPT-SoVITS 저장소 자동 클론
vtts setup --engine gptsovits

# 3. 서버 실행 (각각 다른 포트)
vtts serve Supertone/supertonic-2 --port 8001 --device cuda
vtts serve kevinwang676/GPT-SoVITS-v3 --port 8002 --device cuda
```

> **Supertonic + GPT-SoVITS는 같이 설치해도 충돌하지 않습니다!**

#### 옵션 3: CosyVoice만 (2줄!)

```bash
# 1. vTTS + CosyVoice 의존성 설치 (CosyVoice 코드 내장!)
pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. 서버 실행 (모델 자동 다운로드!)
vtts serve FunAudioLLM/CosyVoice2-0.5B --device cuda --port 8001
```

> **CosyVoice 코드가 vTTS에 내장되어 별도 클론이 필요 없습니다!**

#### 옵션 4: Qwen3-TTS (고품질 TTS)

```bash
# 1. vTTS + Qwen3-TTS 설치
pip install "vtts[qwen3tts] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. CustomVoice (프리셋 스피커, 참조 오디오 불필요)
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --device cuda --port 8001

# 3. Base (Voice Clone, 참조 오디오 필요)
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-Base --device cuda --port 8001
```

> **Qwen3-TTS는 10개 언어 지원, CustomVoice는 9개 프리셋 스피커 제공!**

#### 옵션 5: 모든 엔진 한번에

```bash
# 모든 TTS 엔진 설치
pip install "vtts[all] @ git+https://github.com/bellkjtt/vTTS.git"
```

### Docker (여러 엔진 동시 사용)

```bash
# 개별 실행
docker-compose up -d supertonic   # :8001
docker-compose up -d gptsovits    # :8002 (reference_audio 폴더 필요)
docker-compose up -d cosyvoice    # :8003

# 전체 + Nginx API Gateway
docker-compose --profile gateway up -d  # :8000 (통합 엔드포인트)
```

자세한 내용: [Docker 가이드](DOCKER.md)

### CLI 자동 설치

```bash
# 기본 설치 (Supertonic만)
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# CosyVoice (코드 내장!)
pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"

# GPT-SoVITS (저장소 클론 필요)
pip install "vtts[gptsovits] @ git+https://github.com/bellkjtt/vTTS.git"
vtts setup --engine gptsovits            # GPT-SoVITS 저장소 자동 클론
```

---

## 환경 설정

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
vTTS Environment Diagnosis

✓ Python: 3.10.12
✓ numpy: 1.26.4
✓ onnxruntime: 1.16.0 (CUDA 지원)
  Providers: CUDAExecutionProvider, CPUExecutionProvider
✓ PyTorch: 2.1.0 (CUDA 12.1)
  GPU: NVIDIA GeForce RTX 4090
✓ vTTS: 설치됨

모든 환경이 정상입니다!
```

### Kaggle/Colab에서

```bash
# Supertonic (가장 간단!)
!pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"
!vtts serve Supertone/supertonic-2 --device cuda &

# CosyVoice (2줄!)
!pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"
!vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --device cuda &
```

---

## 서버 실행

### Supertonic (빠른 TTS)
```bash
vtts serve Supertone/supertonic-2
vtts serve Supertone/supertonic-2 --device cuda --port 8000
```

### CosyVoice (다국어 음성 클로닝)

```bash
# 1. vTTS + CosyVoice 의존성 설치 (한 줄!)
pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. 서버 실행 (모델 자동 다운로드!)
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --device cuda --port 8001
```

**참고:**
- CosyVoice 코드가 **vTTS에 내장**되어 있어 별도 클론 불필요!
- 첫 실행 시 HuggingFace에서 **자동으로 모델을 다운로드**합니다 (~2 GB)
- **9개 언어** 지원: 한국어, 영어, 중국어, 일본어, 광동어, 스페인어, 프랑스어, 독일어, 포르투갈어
- **18+ 중국어 방언** 지원

### GPT-SoVITS (음성 클로닝)

```bash
# GPT-SoVITS 저장소 설치 (위의 "방법 2" 참고)
vtts setup --engine gptsovits

# 서버 실행 (pretrained 모델 자동 다운로드됨!)
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

**참고:**
- 첫 실행 시 [HuggingFace](https://huggingface.co/kevinwang676/GPT-SoVITS-v3/tree/main/GPT_SoVITS/pretrained_models)에서 **자동으로 pretrained 모델을 다운로드**합니다 (~2.9 GB)
- 모델은 `~/.cache/huggingface/` 에 캐시되며, 이후 재사용됩니다

### Qwen3-TTS (고품질 다국어 TTS)

```bash
# CustomVoice: 프리셋 스피커 사용 (참조 오디오 불필요)
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --device cuda --port 8003

# Base: Voice Clone (참조 오디오 필요)
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-Base --device cuda --port 8003

# 고품질 1.7B 모델 (더 자연스러운 음성)
vtts serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda --port 8003
```

**참고:**
- **10개 언어 지원**: 한국어, 영어, 중국어, 일본어, 독일어, 프랑스어, 러시아어, 포르투갈어, 스페인어, 이탈리아어
- **CustomVoice 스피커**: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, **Sohee** (한국어)
- **참조 오디오 캐싱**: 동일 참조 오디오 재사용 시 지연 없음

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

## Python 사용

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

### 음성 클로닝 (CosyVoice)
```python
from vtts import VTTSClient

# CosyVoice 클라이언트 (9개 언어 지원!)
client = VTTSClient("http://localhost:8001")

audio = client.tts(
    text="안녕하세요, CosyVoice로 생성한 음성입니다.",
    model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    voice="reference",
    language="ko",  # ko, en, zh, ja, yue, es, fr, de, pt
    reference_audio="./samples/reference.wav",  # 참조 오디오 (제로샷 클로닝)
    reference_text="참조 오디오에서 말하는 내용"
)
audio.save("cosyvoice_output.wav")
```
> **NOTE**: CosyVoice는 9개 언어를 지원하며, 18+ 중국어 방언도 지원합니다!

### 음성 클로닝 (GPT-SoVITS)
```python
from vtts import VTTSClient

# GPT-SoVITS 클라이언트 (참조 오디오 필수!)
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="안녕하세요, 음성 클로닝 테스트입니다.",
    model="kevinwang676/GPT-SoVITS-v3",
    voice="reference",
    language="ko",
    reference_audio="./samples/reference.wav",  # 참조 오디오 (필수!)
    reference_text="참조 오디오에서 말하는 내용",  # 참조 텍스트 (필수!)
    # 품질 조절 파라미터 (선택)
    speed=1.0,                  # 속도 (0.5-2.0)
    top_k=15,                   # Top-K 샘플링 (1-100)
    top_p=1.0,                  # Top-P 샘플링 (0.0-1.0)
    temperature=1.0,            # 다양성 (0.1-2.0, 낮을수록 안정적)
    sample_steps=32,            # 샘플링 스텝 (1-100, 높을수록 품질↑)
    seed=-1,                    # 시드 (-1: 랜덤, 고정값: 재현 가능)
    repetition_penalty=1.35,    # 반복 억제 (1.0-2.0, 높을수록 반복 감소)
    text_split_method="cut5",   # 텍스트 분할 (cut5, four_sentences 등)
    batch_size=1,               # 배치 크기 (1-10)
    fragment_interval=0.3,      # 문장 조각 간 간격 초 (0.0-2.0)
    parallel_infer=True         # 병렬 추론 활성화
)
audio.save("cloned_voice.wav")
```
> **NOTE**: GPT-SoVITS는 `reference_audio`와 `reference_text` 파라미터가 필수입니다!

### Qwen3-TTS (CustomVoice - 프리셋 스피커)
```python
from vtts import VTTSClient

# Qwen3-TTS CustomVoice 클라이언트
client = VTTSClient("http://localhost:8003")

audio = client.tts(
    text="안녕하세요, Qwen3 TTS로 생성한 음성입니다.",
    model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    voice="Sohee",    # 한국어 스피커
    language="ko"
)
audio.save("qwen3_customvoice.wav")
```

**CustomVoice 스피커 목록:**
- 한국어: `Sohee`
- 영어: `Vivian`, `Serena`, `Dylan`, `Eric`, `Ryan`, `Aiden`
- 일본어: `Ono_Anna`
- 중국어: `Uncle_Fu`

### Qwen3-TTS (Base - Voice Clone)
```python
from vtts import VTTSClient

# Qwen3-TTS Base 클라이언트 (Voice Clone)
client = VTTSClient("http://localhost:8003")

audio = client.tts(
    text="안녕하세요, Voice Clone으로 생성한 음성입니다.",
    model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    voice="reference",
    language="ko",
    reference_audio="./samples/reference.wav",  # 참조 오디오 (필수!)
    reference_text="참조 오디오에서 말하는 내용"  # 참조 텍스트 (필수!)
)
audio.save("qwen3_voice_clone.wav")
```

> **NOTE**: Qwen3-TTS Base는 동일 참조 오디오 재사용 시 voice_clone_prompt가 캐싱되어 속도가 향상됩니다!

**파라미터 가이드:**
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `top_k` | 15 | 1-100 | Top-K 샘플링 (낮을수록 보수적) |
| `top_p` | 1.0 | 0.0-1.0 | Nucleus 샘플링 (낮을수록 집중적) |
| `temperature` | 1.0 | 0.1-2.0 | 생성 다양성 (낮을수록 안정적) |
| `sample_steps` | 32 | 1-100 | 샘플링 스텝 (높을수록 품질↑) |
| `seed` | -1 | -1 또는 양수 | 랜덤 시드 (-1: 랜덤) |
| `repetition_penalty` | 1.35 | 1.0-2.0 | 반복 억제 (높을수록 반복 감소) |
| `text_split_method` | cut5 | - | 텍스트 분할 방식 |
| `batch_size` | 1 | 1-10 | 배치 크기 |
| `fragment_interval` | 0.3 | 0.0-2.0 | 문장 간 무음 (초) |
| `parallel_infer` | True | bool | 병렬 추론 |

**시나리오별 추천:**
- **고품질/안정적**: `temperature=0.7, top_p=0.9, sample_steps=40, repetition_penalty=1.5`
- **빠른 생성**: `sample_steps=16, top_k=10, batch_size=2`
- **다양한 결과**: `temperature=1.2, top_k=30, repetition_penalty=1.2`
- **긴 텍스트**: `text_split_method="four_sentences", fragment_interval=0.5`

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

#### Supertonic (가장 빠름)
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "F1", "model": "Supertone/supertonic-2", "language": "en"}' \
  --output supertonic.mp3
```

#### Qwen3-TTS CustomVoice (프리셋 스피커)
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "안녕하세요, Qwen3 TTS입니다.", "voice": "Sohee", "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "language": "ko"}' \
  --output qwen3_customvoice.wav
```

#### Qwen3-TTS Base (Voice Clone)
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요, Voice Clone입니다.",
    "voice": "reference",
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "language": "ko",
    "reference_audio": "/path/to/reference.wav",
    "reference_text": "참조 오디오의 텍스트"
  }' \
  --output qwen3_voice_clone.wav
```

#### GPT-SoVITS (Voice Clone)
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요, GPT-SoVITS입니다.",
    "voice": "reference",
    "model": "kevinwang676/GPT-SoVITS-v3",
    "language": "ko",
    "reference_audio": "/path/to/reference.wav",
    "reference_text": "참조 오디오의 텍스트"
  }' \
  --output gptsovits.wav
```

#### CosyVoice (Voice Clone)
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요, CosyVoice입니다.",
    "voice": "reference",
    "model": "FunAudioLLM/CosyVoice2-0.5B",
    "language": "ko",
    "reference_audio": "/path/to/reference.wav",
    "reference_text": "참조 오디오의 텍스트"
  }' \
  --output cosyvoice.wav
```

---

## Docker

### 포트 구성
| 엔진 | 포트 | GPU 메모리 |
|------|------|-----------|
| Gateway (Nginx) | 8000 | - |
| Supertonic | 8001 | ~1GB |
| GPT-SoVITS | 8002 | ~4GB |
| CosyVoice | 8003 | ~3GB |
| Qwen3-TTS 0.6B | 8004 | ~2GB |
| Qwen3-TTS 1.7B | 8004 | ~4GB |

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

자세한 내용: [Docker 가이드](DOCKER.md)

---

## CLI 명령어

| 명령어 | 설명 |
|--------|------|
| `vtts serve MODEL` | TTS 서버 시작 |
| `vtts doctor` | 환경 진단 |
| `vtts doctor --fix` | 환경 자동 수정 |
| `vtts setup --engine ENGINE` | 엔진별 설치 |
| `vtts list-models` | 지원 모델 목록 |
| `vtts info MODEL` | 모델 정보 |

---

## 아키텍처

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

## 개발 로드맵

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

## 문서

### 시작하기
- [빠른 시작 가이드](docs/QUICKSTART.md)
- [설치 가이드](docs/INSTALL.md)
- [엔진 설정 가이드](docs/ENGINES_SETUP.md)
- [문제 해결 가이드](TROUBLESHOOTING.md)
- [Docker 가이드](DOCKER.md)

### 예제 및 테스트
- [예제 코드](examples/) - [예제 README](examples/README.md)
- [테스트 스위트](tests/) - [테스트 README](tests/README.md)
  - [Kaggle 노트북 (Supertonic)](tests/kaggle/kaggle_supertonic.ipynb)
  - [Kaggle 노트북 (GPT-SoVITS)](tests/kaggle/kaggle_gptsovits.ipynb)
  - [Kaggle 노트북 (CosyVoice)](tests/kaggle/kaggle_cosyvoice.ipynb)

### 개발자 문서
- [개발 문서](docs/) - [문서 README](docs/README.md)
  - [프로젝트 구조](docs/PROJECT_STRUCTURE.md)
  - [프로젝트 현황](docs/PROJECT_STATUS.md)
  - [릴리스 체크리스트](docs/RELEASE_CHECKLIST.md)

### 다국어 문서
- [English](docs/i18n/README_EN.md)
- [中文](docs/i18n/README_ZH.md)
- [日本語](docs/i18n/README_JA.md)

---

## 문제 해결

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

더 많은 문제: [문제 해결 가이드](TROUBLESHOOTING.md)

---

## 라이선스

Apache License 2.0

## 후원

이 프로젝트가 도움이 되셨나요? 

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

## 감사의 말

- [vLLM](https://github.com/vllm-project/vllm) - 아키텍처 영감
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) - Alibaba
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
