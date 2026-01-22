# vTTS Examples

[🏠 홈으로 돌아가기](../README.md) | [🧪 테스트](../tests/README.md) | [📚 개발 문서](../docs/README.md)

vTTS API 사용 예제 모음입니다.

## 📚 예제 목록

### Python 예제

#### 1. [basic_usage.py](basic_usage.py)
**기본 TTS 사용법**
```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")
audio = client.tts(text="안녕하세요", voice="F1")
audio.save("output.wav")
```

**특징:**
- 가장 간단한 TTS 사용법
- 다양한 음성 스타일 테스트
- 언어별 예제

---

#### 2. [combined_tts_stt.py](combined_tts_stt.py)
**TTS + STT 통합 사용**
```python
# TTS로 음성 생성
audio = client.tts(text="음성 인식 테스트")

# STT로 음성 인식
text = client.stt(audio_file="test.wav")
```

**특징:**
- TTS와 STT를 함께 사용
- 음성 생성 → 인식 파이프라인
- 정확도 테스트

---

#### 3. [stt_usage.py](stt_usage.py)
**음성 인식 (STT)**
```python
# 파일에서 음성 인식
result = client.stt(audio_file="audio.wav")

# 스트리밍 음성 인식
for partial in client.stt_stream(audio_stream):
    print(partial)
```

**특징:**
- Faster-Whisper 기반
- 파일 및 스트리밍 지원
- 다국어 인식

---

#### 4. [openai_compatible.py](openai_compatible.py)
**OpenAI SDK 호환 API**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# OpenAI 스타일 TTS
response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="Hello, world!"
)
response.stream_to_file("speech.mp3")
```

**특징:**
- OpenAI API와 완전 호환
- 기존 OpenAI 코드 재사용 가능
- `openai` 라이브러리 사용

---

### Shell 예제

#### 5. [curl_examples.sh](curl_examples.sh)
**cURL을 이용한 API 호출**
```bash
# TTS API
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Supertone/supertonic-2",
    "input": "Hello!",
    "voice": "F1"
  }' \
  --output speech.mp3

# STT API
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=base
```

**특징:**
- HTTP API 직접 호출
- 언어 무관 사용 가능
- CI/CD 통합 예제

---

## 🚀 빠른 시작

### 1. 서버 시작

```bash
# Supertonic
vtts serve Supertone/supertonic-2 --device cuda

# GPT-SoVITS
vtts serve kevinwang676/GPT-SoVITS-v3 --port 8002

# Docker
docker-compose up -d
```

### 2. 예제 실행

```bash
# Python 예제
python examples/basic_usage.py
python examples/openai_compatible.py

# cURL 예제
bash examples/curl_examples.sh
```

---

## 📖 엔진별 사용법

### Supertonic (멀티링구얼)
```python
client = VTTSClient("http://localhost:8000")

# 한국어
audio = client.tts(text="안녕하세요", language="ko", voice="F1")

# 영어
audio = client.tts(text="Hello", language="en", voice="M1")

# 속도 조절
audio = client.tts(text="빠르게", speed=1.5)
```

### GPT-SoVITS (음성 클로닝)
```python
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="클로닝된 음성입니다.",
    model="kevinwang676/GPT-SoVITS-v3",
    language="ko",
    reference_audio="./reference.wav",  # 참조 오디오 (필수)
    reference_text="참조 오디오 내용",    # 참조 텍스트 (필수)
    # 품질 조절
    top_k=15,
    top_p=1.0,
    temperature=1.0,
    sample_steps=32
)
```

### CosyVoice (Zero-shot)
```python
client = VTTSClient("http://localhost:8003")

audio = client.tts(
    text="CosyVoice 테스트",
    model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    language="ko",
    reference_audio="./reference.wav",
    reference_text="참조 음성",
    speed=1.0
)
```

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [설치 가이드](../INSTALL.md)
- [빠른 시작](../QUICKSTART.md)
- [Docker 가이드](../DOCKER.md)
- [문제 해결](../TROUBLESHOOTING.md)
- [Kaggle 테스트](../tests/kaggle/)

## 🤝 기여

새로운 예제를 추가하고 싶으시면 PR을 보내주세요!

1. 예제 코드 작성
2. 주석과 설명 추가
3. 이 README 업데이트
4. PR 제출
