# vTTS 빠른 시작 가이드

## 5분 안에 시작하기

### 1. 설치

```bash
pip install vtts
```

### 2. 서버 시작

```bash
# Supertonic-2 서버 시작 (초고속 한국어 TTS)
vtts serve Supertone/supertonic-2

# 또는 CosyVoice3 (다국어 지원)
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512

# 또는 GPT-SoVITS (Few-shot 음성 복제)
vtts serve kevinwang676/GPT-SoVITS-v3
```

서버가 시작되면 다음과 같이 표시됩니다:

```
🚀 Starting vTTS Server
Model: Supertone/supertonic-2
Host: 0.0.0.0:8000
Device: cuda
Engine: SupertonicEngine

✓ Server starting...
OpenAI compatible API: http://0.0.0.0:8000/v1
Docs: http://0.0.0.0:8000/docs
```

### 3. 사용하기

#### Python

```python
from vtts import VTTSClient

client = VTTSClient()
audio = client.tts("안녕하세요, vTTS입니다!")
audio.save("output.mp3")
```

#### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.audio.speech.create(
    model="auto",
    voice="default",
    input="안녕하세요!"
)
response.stream_to_file("output.mp3")
```

#### cURL

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "안녕하세요!",
    "language": "ko"
  }' \
  --output speech.mp3
```

## 모델 비교

| 모델 | 언어 | 속도 | Zero-shot | 특징 |
|------|------|------|-----------|------|
| **Supertonic-2** | 5개 | ⚡⚡⚡ | ❌ | 초고속, 온디바이스 |
| **CosyVoice3** | 9개 | ⚡⚡ | ✅ | 다국어, 방언 지원 |
| **GPT-SoVITS** | 5개 | ⚡ | ✅ | Few-shot 복제 |

## 다음 단계

- 📖 [전체 문서](README.md)
- 🔧 [설치 가이드](INSTALL.md)
- 💡 [예제 코드](examples/)
- 🚀 [API 문서](http://localhost:8000/docs)
