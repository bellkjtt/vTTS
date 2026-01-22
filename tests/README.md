# vTTS Tests

vTTS 프로젝트의 테스트 및 예제 모음입니다.

## 📂 디렉토리 구조

```
tests/
├── kaggle/              # Kaggle 노트북 테스트
│   ├── kaggle_supertonic.ipynb    # Supertonic 테스트
│   ├── kaggle_gptsovits.ipynb     # GPT-SoVITS 테스트
│   └── kaggle_cosyvoice.ipynb     # CosyVoice 테스트
├── unit/                # 단위 테스트
└── integration/         # 통합 테스트
```

## 🧪 Kaggle 노트북

### Supertonic 테스트
- **파일**: `kaggle/kaggle_supertonic.ipynb`
- **목적**: Supertonic-2 멀티링구얼 TTS 테스트
- **GPU**: T4 x2 권장
- **소요 시간**: ~10분

### GPT-SoVITS 테스트
- **파일**: `kaggle/kaggle_gptsovits.ipynb`
- **목적**: GPT-SoVITS v3 음성 클로닝 테스트
- **GPU**: T4 x2 필수
- **소요 시간**: ~15-20분
- **특징**: Zero-shot voice cloning

### CosyVoice 테스트
- **파일**: `kaggle/kaggle_cosyvoice.ipynb`
- **목적**: CosyVoice3 Zero-shot TTS 테스트
- **GPU**: T4 x2 필수
- **소요 시간**: ~15-20분
- **특징**: 9개 언어 지원, 고품질 음성

## 🚀 사용법

### Kaggle에서 실행

1. **Kaggle 노트북 생성**
   - New Notebook 클릭
   - Settings → Accelerator → GPU T4 x2 선택

2. **노트북 업로드**
   - Upload Notebook 클릭
   - `tests/kaggle/` 내 원하는 노트북 선택

3. **실행**
   - Run All 클릭
   - 각 셀을 순차적으로 실행

### 로컬에서 테스트

```bash
# 단위 테스트 (향후 추가 예정)
pytest tests/unit/

# 통합 테스트 (향후 추가 예정)
pytest tests/integration/
```

## 📝 노트북 구조

모든 Kaggle 노트북은 다음 구조를 따릅니다:

1. **환경 설정 및 설치**: vTTS + 엔진 설치
2. **참조 오디오 생성**: Supertonic으로 참조 오디오 생성
3. **서버 시작**: vTTS 서버 백그라운드 실행
4. **테스트 실행**: 음성 클로닝 및 TTS 테스트
5. **Cleanup**: 서버 종료

## 🔧 문제 해결

### numpy 호환성 문제
```python
# 노트북 내에서 자동으로 처리됨
subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "-q"])
subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0", "-q"])
```

### CUDA 미지원
```python
# onnxruntime-gpu 재설치
subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y", "-q"])
subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu>=1.16.0", "-q"])
```

### 서버 시작 실패
```python
# 로그 확인
with open("server.log", "r") as f:
    print(f.read())
```

## 📚 참고 문서

- [vTTS README](../README.md)
- [Docker 가이드](../DOCKER.md)
- [문제 해결](../TROUBLESHOOTING.md)

## 🤝 기여

테스트 추가를 원하시면 PR을 보내주세요!

1. 새 테스트 작성
2. 문서화
3. PR 제출
