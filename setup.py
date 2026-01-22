"""Setup script for vTTS

의존성 관리 전략:
- 기본 의존성: 모든 엔진 공통 (torch, fastapi 등)
- 엔진별 extras: 충돌 방지를 위해 분리
- 버전 pinning: 호환성 보장

Docker 사용 권장:
- 여러 엔진 동시 사용 시 Docker 컨테이너 분리 권장
- 각 엔진별 Dockerfile 제공: docker/Dockerfile.{engine}
"""

from setuptools import setup, find_packages
from pathlib import Path

# README 읽기
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# ============================================================
# 공통 의존성 (버전 범위 명시로 충돌 최소화)
# ============================================================
CORE_DEPS = [
    # Web Framework
    "fastapi>=0.109.0,<1.0.0",
    "uvicorn[standard]>=0.27.0,<1.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "pydantic-settings>=2.0.0,<3.0.0",
    
    # HTTP Client
    "httpx>=0.26.0,<1.0.0",
    "python-multipart>=0.0.9,<1.0.0",
    
    # CLI & Logging
    "click>=8.1.0,<9.0.0",
    "rich>=13.0.0,<14.0.0",
    "loguru>=0.7.0,<1.0.0",
    
    # ML Core (버전 범위 넓게 - 엔진별 호환성)
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0,<2.0.0",  # numpy 2.0 호환성 이슈 방지
    
    # Audio Processing
    "soundfile>=0.12.0,<1.0.0",
    "librosa>=0.10.0,<1.0.0",
    
    # HuggingFace
    "huggingface-hub>=0.20.0,<1.0.0",
    
    # OpenAI Compatible API
    "openai>=1.0.0,<2.0.0",
]

# STT 의존성 (기본 포함)
STT_DEPS = [
    "faster-whisper>=1.0.0,<2.0.0",
]

# ============================================================
# 엔진별 의존성 (격리)
# ============================================================

# Supertonic: ONNX 기반, 가벼움
# 기본: GPU 지원 (CPU에서도 동작, GPU 있으면 자동 사용)
SUPERTONIC_DEPS = [
    "onnxruntime-gpu>=1.16.0,<2.0.0",
]

# CPU 전용 (GPU 드라이버 없는 환경)
SUPERTONIC_CPU_DEPS = [
    "onnxruntime>=1.16.0,<2.0.0",
]

# CosyVoice: ModelScope 기반
COSYVOICE_DEPS = [
    "modelscope>=1.9.0,<2.0.0",
    "HyperPyYAML>=1.2.0,<2.0.0",
    "conformer>=0.3.0,<1.0.0",
    "wetext>=0.0.4",
    "x-transformers>=2.11.0,<3.0.0",
    "diffusers>=0.29.0,<1.0.0",
]

# GPT-SoVITS: 가장 복잡한 의존성
# Note: GPT-SoVITS 저장소 클론 필요 (pip install로 설치 안됨)
# git clone https://github.com/RVC-Boss/GPT-SoVITS.git
GPTSOVITS_DEPS = [
    # Transformers (버전 범위 매우 중요!)
    "transformers>=4.43.0,<4.51.0",
    "peft>=0.10.0,<0.18.0",
    "sentencepiece>=0.1.99",
    
    # Audio Processing
    "funasr>=1.0.27,<2.0.0",
    "pytorch-lightning>=2.4.0",
    
    # Text Processing (G2P, NLP)
    "g2p_en>=2.1.0",
    "pyopenjtalk>=0.4.1",
    "cn2an>=0.5.0",
    "pypinyin>=0.50.0",
    "jieba>=0.42.1",
    "jieba_fast>=0.53",
    "wordsegment>=1.3.1",
    "split-lang>=0.0.3",
    "fast_langdetect>=0.3.1",
    
    # Korean
    "g2pk2>=0.0.3",
    "ko_pron>=1.0.0",
    
    # Model Architecture
    "rotary_embedding_torch>=0.6.0",
    "x_transformers>=1.30.0",
    
    # FFmpeg
    "ffmpeg-python>=0.2.0",
]

# ============================================================
# Setup
# ============================================================
setup(
    name="vtts",
    version="0.1.0-beta",
    author="vTTS Contributors",
    author_email="",
    description="Universal TTS/STT Serving System - vLLM for Speech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bellkjtt/vTTS",
    project_urls={
        "Bug Tracker": "https://github.com/bellkjtt/vTTS/issues",
        "Documentation": "https://github.com/bellkjtt/vTTS",
        "Source Code": "https://github.com/bellkjtt/vTTS",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10,<3.13",  # Python 3.13 아직 미지원
    
    install_requires=CORE_DEPS + STT_DEPS,
    
    extras_require={
        # 개발용
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        
        # ============================================================
        # 개별 엔진 (권장: 하나씩만 설치)
        # ============================================================
        "supertonic": SUPERTONIC_DEPS,  # 기본 GPU 지원
        "supertonic-cpu": SUPERTONIC_CPU_DEPS,  # CPU 전용
        "cosyvoice": COSYVOICE_DEPS,
        "gptsovits": GPTSOVITS_DEPS,
        
        # ============================================================
        # 조합 (충돌 가능성 있음 - Docker 권장)
        # ============================================================
        
        # Supertonic + CosyVoice (비교적 안전)
        "supertonic-cosyvoice": SUPERTONIC_DEPS + COSYVOICE_DEPS,
        
        # 전체 설치 (충돌 가능 - Docker 강력 권장!)
        "all": SUPERTONIC_DEPS + COSYVOICE_DEPS + GPTSOVITS_DEPS,
        
        # CUDA 지원 (하위 호환성)
        "cuda": SUPERTONIC_DEPS,
    },
    
    entry_points={
        "console_scripts": [
            "vtts=vtts.cli:main",
        ],
    },
)
