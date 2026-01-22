# TTS/STT 엔진 설치 가이드

vTTS는 여러 TTS/STT 엔진을 지원합니다. 각 엔진은 독립적으로 설치할 수 있습니다.

## 🚀 빠른 설치

### 모든 엔진 한 번에 설치 (권장)
```bash
# 1. 모든 dependency 설치
pip install "vtts[all]"

# 2. 필요한 엔진 저장소 클론
# CosyVoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
export PYTHONPATH="$PWD/CosyVoice:$PYTHONPATH"

# GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
export PYTHONPATH="$PWD/GPT-SoVITS:$PYTHONPATH"

# 3. 서버 시작 (원하는 모델 선택)
vtts serve Supertone/supertonic-2  # 또는 다른 모델
```

### 개별 엔진 설치
```bash
# Supertonic-2만
pip install "vtts[supertonic]"

# CosyVoice만
pip install "vtts[cosyvoice]"

# GPT-SoVITS만
pip install "vtts[gptsovits]"
```

---

## 🎙️ TTS 엔진

### 1. Supertonic-2 (추천 - 가장 간단) ⭐

**특징**:
- ONNX 기반 경량 TTS
- 5개 언어 지원 (en, ko, es, pt, fr)
- 매우 빠른 추론 속도
- 66M 파라미터

**설치**:
```bash
# vTTS와 함께 설치
pip install "vtts[supertonic]"

# 또는 수동 설치
pip install supertonic
```

**사용 예시**:
```bash
vtts serve Supertone/supertonic-2 --port 8000
```

**음성 스타일**: M1, M2, M3, M4 (남성), F1, F2, F3, F4 (여성)

---

### 2. CosyVoice3 (Zero-shot 지원)

**특징**:
- Zero-shot 다국어 TTS
- 9개 언어, 18+ 중국 방언
- 1.5B 파라미터
- 스트리밍 지원

**설치** (2가지 방법):

**방법 A: 자동 설치 (권장)** ✅
```bash
# 1. vTTS와 CosyVoice dependency 설치
pip install "vtts[cosyvoice]"

# 2. CosyVoice 저장소 클론 (필수)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# 3. PYTHONPATH 설정
export PYTHONPATH="$PWD:$PYTHONPATH"  # Linux/Mac
# Windows: set PYTHONPATH=%CD%;%PYTHONPATH%

# 4. vTTS 서버 시작
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

**방법 B: 수동 설치**
```bash
# 1. CosyVoice 저장소 클론
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 2. 환경 설정
conda create -n cosyvoice python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5

# 3. 의존성 설치
pip install -r requirements.txt

# 4. vTTS 설치
pip install vtts

# 5. 환경 변수 설정 (중요!)
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:$PYTHONPATH"
```

**사용 예시**:
```bash
# CosyVoice 환경에서 실행
conda activate cosyvoice
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

**참조 오디오 사용** (Zero-shot):
```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

audio = client.tts(
    text="안녕하세요, vTTS입니다.",
    model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    language="ko",
    reference_audio="reference.wav",  # 참조 음성
    reference_text="참조 음성 텍스트"
)
```

---

### 3. GPT-SoVITS (Few-shot Voice Cloning)

**특징**:
- Few-shot: 1분 학습 데이터
- Zero-shot: 5초 참조 오디오
- 5개 언어 (zh, en, ja, ko, yue)
- 매우 자연스러운 음성

**설치** (2가지 방법):

**방법 A: 자동 설치 (권장)** ✅
```bash
# 1. vTTS와 GPT-SoVITS dependency 설치
pip install "vtts[gptsovits]"

# 2. GPT-SoVITS 저장소 클론 (필수)
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# 3. PYTHONPATH 설정
export PYTHONPATH="$PWD:$PYTHONPATH"  # Linux/Mac
# Windows: set PYTHONPATH=%CD%;%PYTHONPATH%

# 4. vTTS 서버 시작
vtts serve kevinwang676/GPT-SoVITS-v3
```

**방법 B: 수동 설치**
```bash
# 1. GPT-SoVITS 저장소 클론
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# 2. 환경 설정
conda create -n gptsovits python=3.10
conda activate gptsovits
pip install -r requirements.txt

# 3. vTTS 설치
pip install vtts

# 4. 환경 변수 설정
export PYTHONPATH="$PWD:$PYTHONPATH"
```

**사용 예시**:
```bash
conda activate gptsovits
vtts serve kevinwang676/GPT-SoVITS-v3 --port 8000
```

**참조 오디오 필수**:
```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

audio = client.tts(
    text="안녕하세요, GPT-SoVITS입니다.",
    model="kevinwang676/GPT-SoVITS-v3",
    language="ko",
    reference_audio="voice_sample.wav",  # 5초+ 참조 음성
    reference_text="참조 음성에서 말한 텍스트"  # 필수!
)
```

---

## 🎤 STT 엔진

### Faster-Whisper (기본 포함) ✅

**특징**:
- CTranslate2 기반 고성능 Whisper
- 99개 언어 지원
- GPU 가속 지원
- 타임스탬프, SRT, VTT 지원

**설치**:
```bash
# vTTS 설치 시 자동 포함
pip install vtts

# 또는 명시적 설치
pip install faster-whisper
```

**사용 예시**:
```bash
# TTS와 STT 동시 서빙
vtts serve Supertone/supertonic-2 --stt-model large-v3 --port 8000
```

---

## 🎯 권장 설치 시나리오

### 시나리오 1: 빠른 테스트 (Supertonic + Faster-Whisper)

```bash
# 가장 간단한 설치
pip install "vtts[supertonic]"

# 서버 시작
vtts serve Supertone/supertonic-2 --stt-model large-v3
```

**장점**: 가장 빠르고 간단, 모든 기능 작동

---

### 시나리오 2: 고품질 Zero-shot (CosyVoice3)

```bash
# CosyVoice 설치
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
conda create -n cosyvoice python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
pip install vtts

# 서버 시작
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:$PYTHONPATH"
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --stt-model large-v3
```

**장점**: 최고 품질, zero-shot 지원

---

### 시나리오 3: Voice Cloning (GPT-SoVITS)

```bash
# GPT-SoVITS 설치
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
conda create -n gptsovits python=3.10
conda activate gptsovits
pip install -r requirements.txt
pip install vtts

# 서버 시작
export PYTHONPATH="$PWD:$PYTHONPATH"
vtts serve kevinwang676/GPT-SoVITS-v3 --stt-model large-v3
```

**장점**: 가장 자연스러운 voice cloning

---

## 🐛 문제 해결

### ImportError: supertonic

```bash
pip install supertonic
```

### ImportError: cosyvoice

CosyVoice는 패키지로 설치할 수 없습니다. 반드시 GitHub에서 클론해야 합니다:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### ImportError: GPT_SoVITS

GPT-SoVITS도 GitHub에서 클론 필요:

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### CUDA Out of Memory

큰 모델은 GPU 메모리를 많이 사용합니다:

```bash
# CPU 모드로 실행
vtts serve <model-id> --device cpu

# 또는 작은 모델 사용
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512  # 0.5B 대신 사용
```

---

## 📊 엔진 비교

| 엔진 | 설치 난이도 | 품질 | 속도 | Zero-shot | 언어 수 |
|------|------------|------|------|-----------|---------|
| **Supertonic-2** | ⭐ 쉬움 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | 5 |
| **CosyVoice3** | ⭐⭐⭐ 어려움 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 9 |
| **GPT-SoVITS** | ⭐⭐⭐ 어려움 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | 5 |
| **Faster-Whisper** | ⭐ 쉬움 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | - | 99 |

---

## 💡 다음 단계

1. 원하는 엔진 설치
2. 서버 시작: `vtts serve <model-id>`
3. API 테스트: `curl http://localhost:8000/docs`
4. Python client 사용: [examples/](examples/) 참고

더 자세한 내용은 [README.md](README.md)를 참조하세요.
