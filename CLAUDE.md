# vTTS - Universal TTS/STT Serving System

> **"vLLM for Speech"** - 모든 TTS/STT 모델을 하나의 통합된 인터페이스로

## 핵심 철학

### 1. 단일 명령어 실행 (One-Command Serving)
```bash
# vLLM처럼 모델 ID만으로 즉시 서버 시작
vtts serve kevinwang676/GPT-SoVITS-v3
vtts serve FunAudioLLM/CosyVoice2-0.5B
vtts serve Supertone/supertonic-2
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
vtts serve ResembleAI/chatterbox          # NEW! Chatterbox
```

**원칙:**
- 모델 ID 하나로 모든 설정 자동 완료
- 의존성 자동 설치 (필요시)
- 프리트레인 모델 자동 다운로드
- 최적 디바이스 자동 선택 (`--device auto/cuda/cpu`)

### 2. OpenAI 호환 API (Drop-in Replacement)
```python
# OpenAI SDK로 그대로 사용 가능
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="gpt-sovits-v3",
    input="안녕하세요",
    voice="reference"
)
```

### 3. 모델 전환 용이성 (Hot-Swapping Ready)
```bash
# 다른 모델로 쉽게 전환 - 같은 포트, 같은 API
vtts serve FunAudioLLM/CosyVoice2-0.5B --port 8000
```

## 지원 엔진 (2026.01) - 6개 엔진

| 엔진 | 모델 ID 패턴 | 언어 | 특징 | 의존성 |
|------|-------------|------|------|--------|
| **Supertonic** | `Supertone/*` | ko, en, ja, zh | ONNX, 빠름 | `.[supertonic]` |
| **Qwen3-TTS** | `Qwen/Qwen3-TTS*` | 10개 언어 | Voice Clone, Base | `.[qwen3tts]` |
| **GPT-SoVITS** | `kevinwang676/*` | zh, en, ja, ko, yue | Zero-shot Voice Clone | `.[gptsovits]` |
| **CosyVoice** | `FunAudioLLM/*` | zh, en, ja, ko + 방언 | Zero-shot TTS | `.[cosyvoice]` |
| **Chatterbox** | `ResembleAI/*` | **23개 언어** | Emotion Control, Turbo | `.[chatterbox]` |
| **KaniTTS** 🆕 | `nineninesix/*` | en, de, zh, ko, ar, es | 15+ 스피커, 초고속 | `.[kanitts]` |

### Chatterbox 모델 종류 (Resemble AI)
- **Chatterbox** (500M): English, CFG & Exaggeration control
- **Chatterbox-Multilingual** (500M): 23개 언어 지원 ✅ Korean 테스트 완료
- **Chatterbox-Turbo** (350M): 저지연, Paralinguistic tags ([laugh], [cough])

### KaniTTS 스피커 (NineNineSix) ✅ Korean/English 테스트 완료
- **English**: david, puck, kore, andrew, jenny, simon, katie
- **Korean**: seulgi
- **German**: bert, thorsten
- **Chinese**: mei (Cantonese), ming (Shanghai)
- **Arabic**: karim, nur
- **Spanish**: maria

> ⚠️ **vTTS 환경 요구사항** (v0.1.0+):
> - **Python 3.11** 필수
> - `transformers==4.57.1` (모든 엔진 통합 호환)
> - `torch>=2.6.0` (보안 패치)
> - KaniTTS는 `nemo-toolkit` 대용량 의존성으로 별도 설치 권장

---

## 🔥 모델 전환 및 동시 사용 가이드

### 방법 1: 순차적 모델 전환 (권장 - 단일 GPU)

GPU 메모리가 제한적인 경우, 하나의 모델만 실행:

```bash
# 모델 A 실행
vtts serve Supertone/supertonic-2 --port 8000

# 모델 B로 전환 시 - 기존 프로세스 종료 후 새 모델 시작
pkill -f "vtts.cli serve"
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8000
```

**API 코드 변경 없이 사용 가능:**
```python
# 같은 코드로 어떤 모델이든 사용
response = requests.post("http://localhost:8000/v1/audio/speech", json={
    "input": "안녕하세요",
    "voice": "F1"  # 또는 "Sohee", "clone" 등 모델에 맞는 voice
})
```

### 방법 2: 다중 모델 동시 실행 (다른 포트)

충분한 GPU 메모리가 있는 경우:

```bash
# 각 모델을 다른 포트에서 실행
vtts serve Supertone/supertonic-2 --port 8001 --device cuda &
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8002 --device cuda &
vtts serve kevinwang676/GPT-SoVITS-v3 --port 8003 --device cuda &
vtts serve FunAudioLLM/CosyVoice2-0.5B --port 8004 --device cuda &
vtts serve ResembleAI/chatterbox --port 8005 --device cuda &
```

**Python에서 모델 선택:**
```python
MODELS = {
    "supertonic": "http://localhost:8001",
    "qwen3": "http://localhost:8002", 
    "gptsovits": "http://localhost:8003",
    "cosyvoice": "http://localhost:8004",
    "chatterbox": "http://localhost:8005",
}

def synthesize(text, model_name="supertonic"):
    url = f"{MODELS[model_name]}/v1/audio/speech"
    return requests.post(url, json={"input": text, "voice": "F1"})
```

### 방법 3: CPU/GPU 혼합 실행

```bash
# 가벼운 모델은 CPU, 무거운 모델은 GPU
vtts serve Supertone/supertonic-2 --port 8001 --device cpu &    # ONNX, CPU 빠름
vtts serve Qwen/Qwen3-TTS-12Hz-0.6B-Base --port 8002 --device cuda &  # GPU 필요
```

### 방법 4: Docker Compose로 분리

```yaml
# docker-compose.yml
services:
  supertonic:
    image: vtts:supertonic
    ports: ["8001:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  
  qwen3tts:
    image: vtts:qwen3tts
    ports: ["8002:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

---

## 📦 의존성 분리 설치

각 엔진은 독립적인 의존성을 가집니다. **충돌 방지를 위해 필요한 엔진만 설치하세요:**

```bash
# 개별 설치 (권장)
pip install -e ".[supertonic]"   # Supertonic만 (가장 가벼움)
pip install -e ".[qwen3tts]"     # Qwen3-TTS만
pip install -e ".[gptsovits]"    # GPT-SoVITS만
pip install -e ".[cosyvoice]"    # CosyVoice만
pip install -e ".[chatterbox]"   # Chatterbox만 (23개 언어)
pip install -e ".[kanitts]"      # KaniTTS만 (nemo-toolkit 필요, 대용량)

# 전체 설치 (의존성 충돌 가능)
pip install -e ".[all]"
```

### 의존성 충돌 해결

| 문제 | 원인 | 해결책 |
|------|------|--------|
| `xformers` 오류 | torch/xformers 버전 불일치 | `XFORMERS_DISABLED=1` 환경변수 설정 |
| `torch.load` 보안 오류 | transformers가 torch 2.6+ 요구 | `pip install torch>=2.6` |
| `transformers` 버전 충돌 | qwen-tts vs GPT-SoVITS | 별도 환경에서 실행 권장 |

### 권장 환경 구성

**방법 A: 단일 환경 (모든 모델)**
```bash
conda create -n vtts python=3.10
conda activate vtts
pip install torch==2.6.0 torchaudio --index-url https://pypi.org/simple/
pip install transformers==4.57.3
pip install -e ".[all]"
```

**방법 B: 모델별 분리 환경 (충돌 완전 방지)**
```bash
# Supertonic용 (가벼움)
conda create -n vtts-supertonic python=3.10
pip install -e ".[supertonic]"

# Qwen3-TTS용
conda create -n vtts-qwen3 python=3.10
pip install -e ".[qwen3tts]"
pip install transformers==4.57.3
```

---

## 📡 API 사용법

### 엔진별 API 예제

**1. Supertonic (빠름, 내장 음성)**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "안녕하세요", "voice": "F1", "language": "ko"}' \
  --output output.wav
```

**2. Qwen3-TTS CustomVoice (Voice Clone)**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "안녕하세요", "voice": "Sohee", "language": "ko"}' \
  --output output.wav
```

**3. GPT-SoVITS (Reference Audio 필수)**
```bash
# reference_audio는 파일 경로 또는 base64
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요",
    "voice": "clone",
    "language": "ko",
    "reference_audio": "/path/to/ref.wav",
    "reference_text": "참조 오디오의 텍스트"
  }' --output output.wav
```

**4. CosyVoice (Zero-shot Clone)**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요",
    "voice": "clone",
    "reference_audio": "/path/to/ref.wav",
    "reference_text": "참조 텍스트"
  }' --output output.wav
```

**5. Chatterbox (23개 언어, Emotion Control)**
```bash
# English (기본)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ResembleAI/chatterbox",
    "input": "Hello, this is Chatterbox TTS!",
    "voice": "default"
  }' --output output.wav

# Korean (Multilingual 모델)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ResembleAI/chatterbox-multilingual",
    "input": "안녕하세요, 한국어 테스트입니다.",
    "voice": "default",
    "language": "ko"
  }' --output output.wav
```

**6. KaniTTS (15+ 스피커, 초고속)**
```bash
# Korean (seulgi 스피커)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nineninesix/kani-tts-370m",
    "input": "안녕하세요, 카니 TTS 테스트입니다.",
    "voice": "seulgi",
    "language": "ko"
  }' --output output.wav

# English (다양한 스피커)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nineninesix/kani-tts-370m",
    "input": "Hello, this is KaniTTS!",
    "voice": "david"
  }' --output output.wav
```

### Python 클라이언트

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 기본 사용
response = client.audio.speech.create(
    model="supertonic",
    input="안녕하세요",
    voice="F1"
)
response.stream_to_file("output.wav")
```

---

## 🏗️ 프로젝트 구조 (확장 설계)

```
vtts/
├── __init__.py
├── cli.py                    # CLI 인터페이스
├── client.py                 # Python 클라이언트
├── server/                   # FastAPI 서버
│   ├── app.py
│   ├── routes.py
│   ├── models.py
│   └── state.py
├── utils/                    # 유틸리티
│   └── audio.py
└── engines/                  # 엔진 모듈
    ├── __init__.py           # 자동 엔진 로더
    ├── base.py               # BaseTTSEngine 추상 클래스
    ├── registry.py           # 엔진 레지스트리
    ├── stt_base.py           # STT 베이스 클래스
    │
    ├── supertonic.py         # Supertonic 엔진
    ├── qwen3tts.py           # Qwen3-TTS 엔진
    ├── gptsovits.py          # GPT-SoVITS 엔진
    ├── cosyvoice.py          # CosyVoice 엔진
    ├── chatterbox.py         # Chatterbox 엔진
    ├── kanitts.py            # KaniTTS 엔진 (NEW!)
    ├── faster_whisper.py     # STT 엔진
    │
    ├── _supertonic/          # 내장 코드 (필요시)
    ├── _gptsovits/           # GPT-SoVITS 내장 코드
    │   ├── TTS_infer_pack/
    │   ├── module/
    │   └── text/
    └── _cosyvoice/           # CosyVoice 내장 코드
        ├── cli/
        ├── flow/
        └── llm/
```

### 새 엔진 추가 가이드

**1단계: 엔진 파일 생성** (`vtts/engines/new_engine.py`)
```python
from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest
from vtts.engines.registry import register_tts_engine

@register_tts_engine(
    name="new_engine",
    model_patterns=["NewOrg/*", "*new-engine*"]
)
class NewEngine(BaseTTSEngine):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self._sample_rate = 24000
    
    def load_model(self) -> None:
        # 모델 로드 로직
        self.is_loaded = True
    
    def unload_model(self) -> None:
        # 모델 언로드
        self.is_loaded = False
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        # TTS 합성 로직
        audio_data = ...  # numpy array
        return TTSOutput(
            audio=audio_data,
            sample_rate=self._sample_rate
        )
    
    @property
    def supported_languages(self) -> list:
        return ["ko", "en"]
```

**2단계: `__init__.py`에 등록** (`vtts/engines/__init__.py`)
```python
from .new_engine import NewEngine
```

**3단계: `pyproject.toml`에 의존성 추가**
```toml
[project.optional-dependencies]
new_engine = ["some-dependency>=1.0.0"]
```

### 네이밍 컨벤션

| 항목 | 컨벤션 | 예시 |
|------|--------|------|
| 엔진 파일 | `snake_case.py` | `qwen3tts.py`, `chatterbox.py` |
| 엔진 클래스 | `PascalCase + Engine` | `Qwen3TTSEngine`, `ChatterboxEngine` |

---

## 📈 확장 가능한 엔진 관리 (20+ 모델 대비)

### 현재 지원 엔진 (6개)
```
vtts/engines/
├── supertonic.py      # Supertone (ONNX)
├── qwen3tts.py        # Alibaba Qwen3-TTS
├── gptsovits.py       # RVC-Boss GPT-SoVITS
├── cosyvoice.py       # Alibaba CosyVoice
├── chatterbox.py      # Resemble AI Chatterbox
├── kanitts.py         # NineNineSix KaniTTS (NEW!)
└── registry.py        # 자동 엔진 등록
```

### 엔진 자동 등록 시스템

`registry.py`에서 모든 엔진이 자동으로 등록됩니다:

```python
# registry.py의 auto_register_engines()
try:
    from vtts.engines.chatterbox import ChatterboxEngine
    EngineRegistry.register(
        "chatterbox",
        ChatterboxEngine,
        model_patterns=["ResembleAI/*", "*chatterbox*"]
    )
except ImportError as e:
    logger.debug(f"Chatterbox engine not available: {e}")
```

### 20개 이상 엔진 추가 시 권장 구조

```
vtts/engines/
├── __init__.py
├── base.py            # BaseTTSEngine
├── registry.py        # 자동 등록 시스템
│
├── # === 기존 엔진 (5개) ===
├── supertonic.py
├── qwen3tts.py
├── gptsovits.py
├── cosyvoice.py
├── chatterbox.py
│
├── # === 향후 추가 예정 ===
├── f5tts.py           # F5-TTS
├── valle.py           # VALL-E
├── xtts.py            # Coqui XTTS
├── bark.py            # Suno Bark
├── tortoise.py        # Tortoise TTS
├── parler.py          # Parler TTS
├── styletts2.py       # StyleTTS 2
├── voicecraft.py      # VoiceCraft
├── metavoice.py       # MetaVoice
├── fishspeech.py      # Fish Speech
│
├── # === 내장 코드 (필요시) ===
├── _gptsovits/        # 내장 GPT-SoVITS
└── _cosyvoice/        # 내장 CosyVoice
```

### 엔진 추가 체크리스트

새 엔진 추가 시:
1. [ ] `vtts/engines/new_engine.py` 생성
2. [ ] `BaseTTSEngine` 상속 및 필수 메서드 구현
3. [ ] `registry.py`의 `auto_register_engines()`에 등록
4. [ ] `pyproject.toml`에 optional dependency 추가
5. [ ] CLAUDE.md 엔진 테이블 업데이트
6. [ ] Fresh 환경에서 테스트 (CUDA + CPU)
| 내장 코드 폴더 | `_prefix` | `_gptsovits/`, `_cosyvoice/` |
| 레지스트리 이름 | `lowercase` | `qwen3tts`, `gptsovits` |

---

## ✅ 테스트 체크리스트

### 새 엔진 추가 시 필수 테스트

```bash
# 1. 서버 시작 테스트
vtts serve NewOrg/new-model --port 8000 --device cuda

# 2. 헬스 체크
curl http://localhost:8000/health

# 3. TTS 생성 테스트
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "테스트 문장", "voice": "default"}' \
  --output test.wav

# 4. 오디오 검증
python -c "import soundfile as sf; d,r = sf.read('test.wav'); print(f'{r}Hz, {len(d)/r:.2f}s')"
```

### CUDA 테스트 결과 (2026-01-24)

| 모델 | CUDA | 샘플레이트 | 생성 시간 | 특징 |
|------|------|-----------|----------|------|
| Supertonic | ✅ | 44100Hz | ~1s | ONNX |
| Qwen3-TTS 0.6B | ✅ | 24000Hz | ~5s | Voice Clone |
| GPT-SoVITS v3 | ✅ | 24000Hz | ~6s | Zero-shot |
| CosyVoice2 0.5B | ✅ | 24000Hz | ~4s | Zero-shot |
| **Chatterbox** | ✅ | 24000Hz | ~2s | English |
| **Chatterbox Korean** | ✅ | 24000Hz | ~2s | Multilingual |
| **KaniTTS Korean** ✅ | ✅ | 22050Hz | ~1.9s | seulgi 스피커 |
| **KaniTTS English** ✅ | ✅ | 22050Hz | ~2.1s | david 스피커 |

---

## 📝 버전 정책

- **0.x**: 베타 버전, API 변경 가능
- **1.x**: 안정 버전, API 호환성 보장

## 📚 관련 문서

- [설치 가이드](docs/INSTALL.md)
- [빠른 시작](docs/QUICKSTART.md)
- [엔진 설정](docs/ENGINES_SETUP.md)
- [Docker 배포](DOCKER.md)
- [문제 해결](TROUBLESHOOTING.md)
