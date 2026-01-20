# vTTS 설치 가이드

## 요구사항

- Python 3.10 이상
- (선택) CUDA 11.8+ (GPU 사용시)

## 기본 설치

```bash
# 1. 저장소 클론
git clone https://github.com/vtts/vtts.git
cd vTTS

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 기본 패키지 설치
pip install -e .
```

## 엔진별 설치

### Supertonic-2

```bash
# Supertonic 패키지 설치 (실제 패키지명 확인 필요)
pip install supertonic

# 또는 ONNX Runtime만 사용
pip install onnxruntime-gpu  # GPU
pip install onnxruntime       # CPU
```

### CosyVoice3

```bash
# CosyVoice 저장소 클론
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
cd ..

# 또는 직접 설치
pip install cosyvoice  # 패키지가 있다면
```

### GPT-SoVITS

```bash
# GPT-SoVITS 저장소 클론
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
cd ..
```

## 모든 엔진 한번에 설치 (권장하지 않음)

```bash
pip install -e ".[all]"
```

## 개발 환경 설치

```bash
pip install -e ".[dev]"
```

## 설치 확인

```bash
# CLI 확인
vtts --help

# 지원 엔진 확인
vtts list-models

# 특정 모델 정보
vtts info Supertone/supertonic-2
```

## Docker 설치

```bash
# Dockerfile 빌드
docker build -t vtts:latest .

# 실행
docker run -p 8000:8000 vtts:latest serve Supertone/supertonic-2

# GPU 사용
docker run --gpus all -p 8000:8000 vtts:latest serve Supertone/supertonic-2
```

## 문제 해결

### CUDA 관련 오류

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ONNX Runtime 오류

```bash
# ONNX Runtime GPU 버전 재설치
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

### FFmpeg 관련 오류

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```
