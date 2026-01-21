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
- 🔌 **플러그인 아키텍처**: 새로운 엔진 쉽게 추가 가능

## 📦 지원 모델

### TTS (Text-to-Speech)
- ✅ **GPT-SoVITS-v3** - Few-shot 음성 복제
- ✅ **Supertonic-2** - 초고속 온디바이스 TTS (5개 언어: en, ko, es, pt, fr)
- ✅ **CosyVoice3** - Zero-shot 다국어 TTS (9개 언어, 18+ 중국 방언)
- 🔜 **StyleTTS2**, **XTTS-v2**, **Bark**

### STT (Speech-to-Text)
- ✅ **Faster-Whisper** - 초고속 Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

## 🚀 빠른 시작

### 설치

#### 기본 설치
```bash
# GitHub에서 설치 (Supertonic-2 + Faster-Whisper 포함)
pip install git+https://github.com/bellkjtt/vTTS.git
pip install supertonic
```

#### 모든 엔진 설치 (권장)
```bash
# 1. 모든 dependency 설치
pip install "vtts[all] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. 고급 엔진 사용을 위한 저장소 클론 (선택)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
export PYTHONPATH="$PWD/CosyVoice:$PWD/GPT-SoVITS:$PYTHONPATH"
```

#### 개별 엔진 설치
```bash
# Supertonic-2만
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# CosyVoice만  
pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"

# GPT-SoVITS만
pip install "vtts[gptsovits] @ git+https://github.com/bellkjtt/vTTS.git"
```

#### Kaggle에서 테스트
[Kaggle 노트북](kaggle_test_notebook.ipynb) 참고

⚠️ **설치 문제가 있나요?** [문제 해결 가이드](TROUBLESHOOTING.md)를 확인하세요.

### 서버 실행

#### TTS 전용
```bash
# 모델 자동 다운로드 및 서버 시작
vtts serve Supertone/supertonic-2

# 포트 지정
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

#### TTS + STT 동시
```bash
# TTS와 STT를 동시에 서빙
vtts serve Supertone/supertonic-2 --stt-model large-v3

# GPU 지정
vtts serve kevinwang676/GPT-SoVITS-v3 --stt-model large-v3 --device cuda:0
```

### Python 사용
```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

# 음성 생성
audio = client.tts(
    text="안녕하세요, vTTS를 사용해주셔서 감사합니다.",
    model="Supertone/supertonic-2",
    language="ko",
    voice="default"
)

# 파일로 저장
audio.save("output.wav")
```

### OpenAI SDK 호환
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="default",
    input="안녕하세요, 반갑습니다."
)

response.stream_to_file("output.mp3")
```

## 🏗️ 아키텍처

```
vTTS/
├── vtts/
│   ├── __init__.py
│   ├── cli.py                 # CLI 진입점
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI 앱
│   │   ├── routes.py         # API 라우트
│   │   └── models.py         # Pydantic 모델
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py           # 베이스 엔진 인터페이스
│   │   ├── registry.py       # 엔진 레지스트리
│   │   ├── gptsovits.py      # GPT-SoVITS 엔진
│   │   ├── supertonic.py     # Supertonic 엔진
│   │   └── cosyvoice.py      # CosyVoice 엔진
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py         # 모델 로더
│   │   └── cache.py          # 모델 캐시 관리
│   └── utils/
│       ├── __init__.py
│       ├── audio.py          # 오디오 처리
│       └── hf.py             # Huggingface 유틸
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## 🔧 개발 로드맵

- [x] 프로젝트 구조 설계
- [ ] 베이스 엔진 인터페이스 구현
- [ ] Supertonic-2 엔진 구현
- [ ] CosyVoice3 엔진 구현
- [ ] GPT-SoVITS 엔진 구현
- [ ] FastAPI 서버 구현
- [ ] OpenAI 호환 API
- [ ] CLI 구현
- [ ] 모델 자동 다운로드
- [ ] 스트리밍 지원
- [ ] 배치 추론 최적화
- [ ] Docker 이미지

## 📚 문서

- [빠른 시작 가이드](QUICKSTART.md)
- [문제 해결 가이드](TROUBLESHOOTING.md) - 500 에러, 설치 문제 등
- [Kaggle 테스트 노트북](kaggle_test_notebook.ipynb)
- [예제 코드](examples/)

## 📝 라이선스

MIT License

## 💖 후원

이 프로젝트가 도움이 되셨나요? 

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

후원해주시면 프로젝트 개발에 큰 도움이 됩니다!

## 🙏 감사의 말

- [vLLM](https://github.com/vllm-project/vllm) - 아키텍처 영감
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
