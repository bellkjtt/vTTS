# GitHub 설정 및 배포 가이드

## 📦 v0.1.0 릴리스 준비

### 1. GitHub 저장소 생성

```bash
# 1. GitHub에서 새 저장소 생성
# Repository name: vTTS
# Description: Universal TTS/STT Serving System - vLLM for Speech
# Public repository
# Add README: No (이미 있음)
# Add .gitignore: No (이미 있음)
# License: MIT

# 2. 로컬 Git 초기화
cd c:\Users\Administer\Downloads\live2d\vTTS
git init
git add .
git commit -m "Initial commit: vTTS v0.1.0"

# 3. GitHub 저장소 연결
git remote add origin https://github.com/YOUR_USERNAME/vTTS.git
git branch -M main
git push -u origin main
```

### 2. GitHub 스폰서 설정

#### `.github/FUNDING.yml` 수정

```yaml
# 스폰서 옵션 (하나 이상 선택)
github: YOUR_GITHUB_USERNAME  # GitHub Sponsors
ko_fi: YOUR_KOFI_USERNAME     # Ko-fi
patreon: YOUR_PATREON_NAME    # Patreon
```

#### GitHub Sponsors 활성화

1. GitHub 프로필 → Settings → Sponsors
2. "Set up GitHub Sponsors" 클릭
3. 은행 정보 등록
4. 스폰서 티어 설정:
   - $5/month: ☕ Coffee Supporter
   - $25/month: 🚀 Bronze Sponsor
   - $100/month: 💎 Silver Sponsor
   - $500/month: 🏆 Gold Sponsor

### 3. 릴리스 태그 생성

```bash
# v0.1.0 태그 생성
git tag -a v0.1.0 -m "Release v0.1.0: Initial release with TTS/STT support"
git push origin v0.1.0
```

### 4. GitHub Release 생성

GitHub 웹사이트에서:

1. Releases → Create a new release
2. Tag: `v0.1.0`
3. Release title: `vTTS v0.1.0 - Initial Release`
4. 설명:

```markdown
# vTTS v0.1.0 - Initial Release 🎉

**vLLM for Speech** - Universal TTS/STT Serving System

## ✨ Features

### TTS (Text-to-Speech)
- 🎙️ **Supertonic-2** - 초고속 온디바이스 TTS
- 🗣️ **CosyVoice3** - Zero-shot 다국어 TTS
- 🎵 **GPT-SoVITS** - Few-shot 음성 복제

### STT (Speech-to-Text)
- 🎤 **Faster-Whisper** - 고성능 음성 인식
- 🌍 99개 언어 지원
- 📊 타임스탬프 & 자막 생성

## 🚀 Quick Start

```bash
# 설치
pip install git+https://github.com/YOUR_USERNAME/vTTS.git

# TTS 서버 시작
vtts serve Supertone/supertonic-2

# TTS + STT 동시
vtts serve Supertone/supertonic-2 --stt-model large-v3
```

## 🌐 OpenAI API Compatible

완전한 OpenAI TTS & Whisper API 호환

## 📚 Documentation

- [README](https://github.com/YOUR_USERNAME/vTTS)
- [Quick Start Guide](QUICKSTART.md)
- [Installation Guide](INSTALL.md)

## 🙏 Support

이 프로젝트가 도움이 되셨다면 스폰서를 고려해주세요!

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink)](https://github.com/sponsors/YOUR_USERNAME)
```

5. Publish release 클릭

### 5. PyPI 배포 (선택적)

```bash
# 빌드 도구 설치
pip install build twine

# 패키지 빌드
python -m build

# TestPyPI에 업로드 (테스트)
python -m twine upload --repository testpypi dist/*

# PyPI에 업로드 (실제)
python -m twine upload dist/*
```

이후 사용자는 다음과 같이 설치 가능:
```bash
pip install vtts
```

### 6. README 배지 추가

README.md 상단에 추가:

```markdown
# vTTS

[![GitHub release](https://img.shields.io/github/v/release/YOUR_USERNAME/vTTS)](https://github.com/YOUR_USERNAME/vTTS/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/YOUR_USERNAME)](https://github.com/sponsors/YOUR_USERNAME)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/vTTS?style=social)](https://github.com/YOUR_USERNAME/vTTS)
```

## 📊 설치 방법 정리

### GitHub에서 직접 설치 (현재)

```bash
pip install git+https://github.com/YOUR_USERNAME/vTTS.git
```

### 특정 버전 설치

```bash
pip install git+https://github.com/YOUR_USERNAME/vTTS.git@v0.1.0
```

### 개발 버전 설치

```bash
git clone https://github.com/YOUR_USERNAME/vTTS.git
cd vTTS
pip install -e .
```

### PyPI 설치 (배포 후)

```bash
pip install vtts
```

## 🧪 Kaggle에서 테스트

### 방법 1: GitHub에서 직접 설치

노트북 첫 셀:
```python
!pip install git+https://github.com/YOUR_USERNAME/vTTS.git
```

### 방법 2: Kaggle Dataset으로 업로드

1. vTTS 폴더를 zip으로 압축
2. Kaggle Datasets에 업로드
3. 노트북에서 사용:

```python
!pip install /kaggle/input/vtts/vTTS.zip
```

## 📱 홍보 전략

### 1. Reddit
- r/MachineLearning
- r/LocalLLaMA
- r/Python

### 2. Twitter/X
해시태그: #TTS #STT #AI #OpenSource

### 3. Hacker News
Show HN: vTTS - vLLM for Speech

### 4. 한국 커뮤니티
- GeekNews
- MLOps Korea
- AI Korea

## 🎯 로드맵

- [ ] v0.1.0 릴리스
- [ ] PyPI 배포
- [ ] Docker Hub 배포
- [ ] GitHub Actions CI/CD
- [ ] 문서 사이트 (MkDocs)
- [ ] v0.2.0: 스트리밍 지원
- [ ] v0.3.0: 배치 추론 최적화
