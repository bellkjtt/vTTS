# vTTS - Universal TTS/STT Serving System

> **"vLLM for Speech"** - 모든 TTS/STT 모델을 하나의 통합된 인터페이스로

## 핵심 철학

### 1. 단일 명령어 실행 (One-Command Serving)
```bash
# vLLM처럼 모델 ID만으로 즉시 서버 시작
vtts serve kevinwang676/GPT-SoVITS-v3
vtts serve FunAudioLLM/CosyVoice2-0.5B
vtts serve Supertone/supertonic-v2
```

**원칙:**
- 모델 ID 하나로 모든 설정 자동 완료
- 의존성 자동 설치 (필요시)
- 프리트레인 모델 자동 다운로드
- 최적 디바이스 자동 선택

### 2. OpenAI 호환 API (Drop-in Replacement)
```python
# OpenAI SDK로 그대로 사용 가능
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")
response = client.audio.speech.create(
    model="gpt-sovits-v3",
    input="안녕하세요",
    voice="reference"
)
```

**원칙:**
- OpenAI TTS API 100% 호환
- 기존 코드 변경 없이 모델만 교체
- 표준화된 에러 응답

### 3. 모델 전환 용이성 (Hot-Swapping Ready)
```bash
# 다른 모델로 쉽게 전환
vtts serve FunAudioLLM/CosyVoice2-0.5B --port 8000
# 같은 API로 사용, 코드 변경 불필요
```

**원칙:**
- 모든 엔진이 동일한 API 인터페이스 구현
- 엔진별 특수 파라미터는 `extra_params`로 전달
- 모델 정보 API로 지원 기능 확인 가능

### 4. 실제 동작 검증 (Real Functionality Test)
모든 기능은 실제로 동작해야 합니다:

**필수 테스트 항목:**
- [ ] wav 파일 생성 및 재생 가능 여부
- [ ] API 응답 시간 측정
- [ ] 다양한 텍스트 입력 처리
- [ ] 모든 파라미터 동작 확인
- [ ] 에러 처리 및 복구
- [ ] 스트리밍 지원 (해당 시)

## 지원 엔진

| 엔진 | 모델 ID 패턴 | 언어 | 특징 |
|------|-------------|------|------|
| GPT-SoVITS | `kevinwang676/*`, `*gptsovits*` | zh, en, ja, ko, yue | Voice cloning |
| CosyVoice | `FunAudioLLM/*`, `*cosyvoice*` | zh, en, ja, ko + 방언 | Zero-shot TTS |
| Supertonic | `Supertone/*`, `*supertonic*` | ko, en, ja, zh | 고품질 한국어 |
| Qwen3-TTS | `Qwen/Qwen3-TTS*` | 10개 언어 | 스트리밍, Voice Design |

## API 사용법

### TTS (Text-to-Speech)

```bash
# 기본 사용
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-sovits-v3",
    "input": "안녕하세요, 반갑습니다.",
    "voice": "reference",
    "reference_audio": "/path/to/reference.wav"
  }' \
  --output output.wav

# 스트리밍
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "안녕하세요", "stream": true}' \
  --output output.wav
```

### Python 클라이언트

```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")

# 간단한 사용
audio = client.synthesize(
    text="안녕하세요",
    reference_audio="reference.wav"
)
audio.save("output.wav")

# 상세 설정
audio = client.synthesize(
    text="안녕하세요",
    reference_audio="reference.wav",
    language="ko",
    speed=1.0,
    extra_params={
        "temperature": 0.7,
        "top_k": 15
    }
)
```

## 개발 원칙

### 1. 모듈화
- 각 엔진은 독립적으로 동작
- 공통 인터페이스 준수 (`BaseTTSEngine`)
- 엔진별 의존성 분리

### 2. 테스트 우선
- 모든 기능에 대한 통합 테스트
- 실제 wav 생성 검증
- CI/CD 파이프라인

### 3. 문서화
- API 문서 자동 생성 (FastAPI)
- 엔진별 사용 가이드
- 예제 코드 제공

### 4. 에러 처리
- 명확한 에러 메시지
- 복구 가능한 상태 유지
- 로깅 및 디버깅 지원

## 디렉토리 구조

```
vtts/
├── engines/           # TTS 엔진 구현
│   ├── base.py       # 공통 인터페이스
│   ├── gptsovits.py  # GPT-SoVITS 엔진
│   ├── cosyvoice.py  # CosyVoice 엔진
│   ├── supertonic.py # Supertonic 엔진
│   └── _gptsovits/   # 내장 GPT-SoVITS 코드
│       ├── f5_tts/   # F5-TTS (v3 필요)
│       ├── eres2net/ # Speaker verification
│       ├── AR/       # Autoregressive 모델
│       └── ...
├── server/           # FastAPI 서버
├── client.py         # Python 클라이언트
└── cli.py           # CLI 인터페이스
```

## 기여 가이드

새 엔진 추가 시:
1. `BaseTTSEngine` 상속
2. 필수 메서드 구현: `load_model()`, `synthesize()`, `unload_model()`
3. 엔진 등록: `@register_tts_engine` 데코레이터
4. 테스트 케이스 작성
5. 문서화

## 버전 정책

- 0.x: 베타 버전, API 변경 가능
- 1.x: 안정 버전, API 호환성 보장
