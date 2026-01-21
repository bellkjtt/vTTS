# vTTS 문제 해결 가이드 (Troubleshooting)

## 🔥 일반적인 문제와 해결방법

### 1. Kaggle/Colab에서 500 Internal Server Error

#### 증상
```python
HTTPStatusError: Server error '500 Internal Server Error' for url 'http://localhost:8000/v1/audio/speech'
```

#### 원인
`vtts[supertonic]` 설치 시 `supertonic` 패키지가 제대로 설치되지 않았습니다.

#### 해결방법

**방법 1: 수동으로 supertonic 설치**
```bash
pip install supertonic>=0.1.0
```

**방법 2: 설치 확인 후 재설치**
```python
# 설치
!pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# 확인
try:
    import supertonic
    print(f"✅ Supertonic installed: {supertonic.__version__}")
except ImportError:
    print("❌ Supertonic not installed. Installing manually...")
    !pip install supertonic>=0.1.0
```

**방법 3: 최신 버전 재설치**
```bash
pip uninstall -y vtts
pip install --no-cache-dir "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"
```

#### 서버 로그 확인
문제가 계속되면 서버 로그를 확인하세요:

```python
# Kaggle/Colab에서
!cat vtts_server.log | tail -n 50
```

---

### 2. 모델 다운로드 실패

#### 증상
```
Failed to download model from Hugging Face
```

#### 해결방법

**HuggingFace Token 설정**
```python
from huggingface_hub import login
login(token="your_hf_token")
```

또는 환경변수 설정:
```bash
export HF_TOKEN="your_hf_token"
```

**캐시 디렉토리 지정**
```bash
vtts serve Supertone/supertonic-2 --cache-dir ./cache
```

---

### 3. CUDA Out of Memory

#### 증상
```
RuntimeError: CUDA out of memory
```

#### 해결방법

**방법 1: CPU 모드 사용**
```bash
vtts serve Supertone/supertonic-2 --device cpu
```

**방법 2: 작은 모델 사용**
```bash
# STT의 경우 작은 모델 선택
vtts serve Supertone/supertonic-2 --stt-model base
vtts serve Supertone/supertonic-2 --stt-model tiny
```

**방법 3: GPU 메모리 정리**
```python
import torch
torch.cuda.empty_cache()
```

---

### 4. Port 8000 이미 사용 중

#### 증상
```
OSError: [Errno 98] Address already in use
```

#### 해결방법

**방법 1: 다른 포트 사용**
```bash
vtts serve Supertone/supertonic-2 --port 8001
```

**방법 2: 기존 프로세스 종료**
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

### 5. 음성이 생성되지 않음 (빈 오디오)

#### 원인
- 잘못된 voice ID
- 지원하지 않는 언어
- 빈 텍스트

#### 해결방법

**사용 가능한 voice 확인**
```bash
curl http://localhost:8000/v1/voices
```

**Python에서 확인**
```python
from vtts import VTTSClient

client = VTTSClient()
voices = client.list_voices()
print(voices)
```

**올바른 voice ID 사용**
- Supertonic-2: M1, M2, M3, M4, F1, F2, F3, F4
- 대소문자 구분 없음 (m1, M1 모두 가능)

---

### 6. ImportError: No module named 'vtts'

#### 해결방법

**설치 확인**
```bash
pip list | grep vtts
```

**재설치**
```bash
pip install --upgrade --force-reinstall "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"
```

---

### 7. 느린 응답 속도

#### 원인
- 첫 실행: 모델 다운로드 및 로딩
- CPU 모드 사용
- 큰 텍스트

#### 해결방법

**GPU 사용 확인**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

**첫 실행 후 속도 개선 확인**
```python
import time

# 첫 실행 (모델 로딩)
start = time.time()
audio1 = client.tts(text="Test", voice="F1")
print(f"First call: {time.time() - start:.2f}s")

# 두 번째 실행 (캐시 사용)
start = time.time()
audio2 = client.tts(text="Test", voice="F1")
print(f"Second call: {time.time() - start:.2f}s")
```

---

### 8. Docker 관련 문제

#### Port 매핑 확인
```bash
docker run -p 8000:8000 vtts:latest vtts serve Supertone/supertonic-2
```

#### 컨테이너 로그 확인
```bash
docker logs <container_id>
```

#### GPU 지원 (NVIDIA)
```bash
docker run --gpus all -p 8000:8000 vtts:latest vtts serve Supertone/supertonic-2
```

---

### 9. OpenAI SDK 호환성 문제

#### vTTS는 OpenAI API와 완전 호환됩니다

**올바른 사용법**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vTTS는 API key 불필요 (dummy 사용)
)

# TTS
response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="Hello world"
)

# STT
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="base",
        file=f
    )
```

---

### 10. 지원 문의

#### GitHub Issues
문제가 해결되지 않으면 GitHub Issues에 보고해주세요:
https://github.com/bellkjtt/vTTS/issues

**보고 시 포함할 정보:**
1. 에러 메시지 전체
2. Python 버전: `python --version`
3. vTTS 버전: `pip show vtts`
4. 운영체제
5. GPU 사용 여부
6. 서버 로그 (vtts_server.log)

#### 환경 정보 수집
```python
import sys
import torch
import vtts

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"vTTS: {vtts.__version__}")
```

---

## 🔍 디버깅 팁

### 1. 로그 레벨 증가
```bash
vtts serve Supertone/supertonic-2 --log-level DEBUG
```

### 2. Health Check
```bash
curl http://localhost:8000/health
```

응답 예시:
```json
{
  "status": "ok",
  "model": "Supertone/supertonic-2",
  "device": "cuda",
  "is_loaded": true
}
```

### 3. 모델 목록 확인
```bash
curl http://localhost:8000/v1/models
```

### 4. API 테스트
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "voice": "F1"}' \
  --output test.mp3
```

---

## 📚 추가 리소스

- [README](README.md) - 기본 사용법
- [QUICKSTART](QUICKSTART.md) - 빠른 시작 가이드
- [GitHub](https://github.com/bellkjtt/vTTS) - 소스 코드
- [Examples](examples/) - 예제 코드
