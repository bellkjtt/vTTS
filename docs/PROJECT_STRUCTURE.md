# vTTS 프로젝트 구조

## 📁 디렉토리 구조

```
vTTS/
├── vtts/                          # 메인 패키지
│   ├── __init__.py               # 패키지 초기화
│   ├── cli.py                    # CLI 진입점
│   ├── client.py                 # Python 클라이언트
│   │
│   ├── engines/                  # TTS 엔진들
│   │   ├── __init__.py
│   │   ├── base.py              # 베이스 엔진 인터페이스
│   │   ├── registry.py          # 엔진 레지스트리
│   │   ├── supertonic.py        # Supertonic-2 엔진
│   │   ├── cosyvoice.py         # CosyVoice3 엔진 (TODO)
│   │   └── gptsovits.py         # GPT-SoVITS 엔진 (TODO)
│   │
│   ├── server/                   # FastAPI 서버
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI 앱
│   │   ├── routes.py            # API 라우트
│   │   ├── models.py            # Pydantic 모델
│   │   └── state.py             # 서버 상태 관리
│   │
│   └── utils/                    # 유틸리티
│       ├── __init__.py
│       └── audio.py             # 오디오 처리
│
├── examples/                      # 예제 코드
│   ├── basic_usage.py           # 기본 사용법
│   ├── openai_compatible.py     # OpenAI SDK 호환
│   └── curl_examples.sh         # cURL 예제
│
├── tests/                         # 테스트 (TODO)
│
├── docs/                          # 문서 (TODO)
│
├── pyproject.toml                # 프로젝트 설정
├── Dockerfile                    # Docker 이미지
├── .gitignore                    # Git 무시 목록
│
├── README.md                     # 메인 README
├── QUICKSTART.md                 # 빠른 시작 가이드
├── INSTALL.md                    # 설치 가이드
└── PROJECT_STRUCTURE.md          # 이 파일
```

## 🔌 아키텍처

### 1. 엔진 시스템

```
BaseTTSEngine (추상 베이스 클래스)
    │
    ├── SupertonicEngine      # Supertonic-2
    ├── CosyVoiceEngine       # CosyVoice3
    └── GPTSoVITSEngine       # GPT-SoVITS
```

각 엔진은 다음을 구현해야 합니다:
- `load_model()`: 모델 로드
- `synthesize()`: 음성 합성
- `supported_languages`: 지원 언어
- `supports_zero_shot`: Zero-shot 지원 여부

### 2. 엔진 레지스트리

```python
# 자동으로 모델 ID에 맞는 엔진 선택
engine_class = EngineRegistry.get_engine_for_model("Supertone/supertonic-2")
# -> SupertonicEngine

engine_class = EngineRegistry.get_engine_for_model("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
# -> CosyVoiceEngine
```

### 3. API 흐름

```
User Request
    ↓
FastAPI Router (/v1/audio/speech)
    ↓
ServerState (전역 엔진 인스턴스)
    ↓
TTSEngine.synthesize()
    ↓
Audio Encoding (mp3, wav, etc)
    ↓
StreamingResponse
```

## 🎯 핵심 개념

### 엔진 독립성
각 TTS 엔진은 독립적으로 구현되며, 공통 인터페이스를 통해 접근합니다.

### 자동 모델 감지
모델 ID를 보고 자동으로 적절한 엔진을 선택합니다.

### OpenAI 호환
OpenAI의 TTS API와 호환되는 엔드포인트를 제공합니다.

### 플러그인 방식
새로운 TTS 엔진을 쉽게 추가할 수 있습니다:

1. `BaseTTSEngine`을 상속하여 엔진 구현
2. `EngineRegistry.register()`로 등록
3. 끝!

## 📝 구현 상태

### ✅ 완료
- [x] 프로젝트 구조 설계
- [x] 베이스 엔진 인터페이스
- [x] 엔진 레지스트리
- [x] Supertonic 엔진 (기본 구현)
- [x] FastAPI 서버
- [x] OpenAI 호환 API
- [x] Python 클라이언트
- [x] CLI 인터페이스
- [x] 예제 코드
- [x] 문서

### 🚧 TODO
- [ ] CosyVoice3 엔진 구현
- [ ] GPT-SoVITS 엔진 구현
- [ ] 스트리밍 지원
- [ ] 배치 추론
- [ ] 모델 자동 다운로드 최적화
- [ ] 테스트 코드
- [ ] CI/CD
- [ ] 성능 벤치마크
- [ ] Docker 이미지 최적화

## 🔧 새 엔진 추가 방법

### 1. 엔진 파일 생성

`vtts/engines/myengine.py`:

```python
from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest

class MyEngine(BaseTTSEngine):
    def load_model(self):
        # 모델 로드 구현
        pass
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        # 음성 합성 구현
        pass
    
    @property
    def supported_languages(self):
        return ["ko", "en"]
    
    # ... 나머지 속성 구현
```

### 2. 레지스트리에 등록

`vtts/engines/registry.py`의 `auto_register_engines()`에 추가:

```python
from vtts.engines.myengine import MyEngine
EngineRegistry.register(
    "myengine",
    MyEngine,
    model_patterns=["myorg/*", "*mymodel*"]
)
```

### 3. 완료!

```bash
vtts serve myorg/mymodel
```

## 📚 참고 자료

- [vLLM 프로젝트](https://github.com/vllm-project/vllm)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [OpenAI TTS API](https://platform.openai.com/docs/guides/text-to-speech)
