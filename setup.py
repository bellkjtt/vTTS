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
# 공통 의존성 (최소한의 공통 패키지만 포함)
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
    
    # Core ML (엔진별로 다른 버전 필요 - 최소 버전만)
    "torch>=2.0.0",  # 엔진별로 오버라이드됨
    "torchaudio>=2.0.0",  # 엔진별로 오버라이드됨
    "numpy>=1.24.0,<2.0.0",  # numpy 2.0 호환성 이슈 방지
    
    # Audio Processing (공통)
    "soundfile>=0.12.0,<1.0.0",
    
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
# 엔진별 의존성 (정확한 버전 고정 - 각 엔진 공식 requirements 기반)
# ============================================================

# Supertonic: ONNX 기반, 가벼움, PyTorch 불필요
# 기본: GPU 지원 (CPU에서도 동작, GPU 있으면 자동 사용)
SUPERTONIC_DEPS = [
    "onnxruntime-gpu>=1.16.0,<1.19.0",  # 1.18.0 권장
]

# CPU 전용 (GPU 드라이버 없는 환경)
SUPERTONIC_CPU_DEPS = [
    "onnxruntime>=1.16.0,<1.19.0",  # 1.18.0 권장
]

# GPT-SoVITS: RVC-Boss/GPT-SoVITS 공식 requirements.txt 기반
# Note: GPT-SoVITS 저장소 클론 필요 (vtts setup --engine gptsovits)
GPTSOVITS_DEPS = [
    # Core ML (GPT-SoVITS 호환)
    "torch>=2.0.0",  # 버전 제약 없음 (유연)
    "torchaudio>=2.0.0",
    
    # Audio
    "librosa==0.10.2",  # 정확히 고정
    
    # Transformers (중요: 4.50 이하!)
    "transformers>=4.43.0,<=4.50.0",  # 4.51+ 호환 불가
    "peft>=0.10.0,<0.18.0",
    "sentencepiece>=0.1.99",
    
    # Audio Processing
    "funasr==1.0.27",  # 정확히 고정
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
    
    # Extras
    "chardet>=4.0.0",
    "PyYAML>=5.0",
    "psutil>=5.0.0",
    "ToJyutping>=0.0.1",
    "opencc>=1.1.0",
    "ctranslate2>=4.0.0,<5.0.0",
    "av>=11.0.0",
]

# CosyVoice: FunAudioLLM/CosyVoice 공식 requirements.txt 기반
# ⚠️ 경고: transformers==4.51.3 필요 (GPT-SoVITS와 충돌!)
# → Docker 사용 강력 권장!
COSYVOICE_DEPS = [
    # Core ML (CosyVoice 정확히 고정)
    "torch==2.3.1",  # 정확히 고정
    "torchaudio==2.3.1",  # 정확히 고정
    
    # Transformers (중요: 4.51.3 필요!)
    "transformers==4.51.3",  # GPT-SoVITS와 충돌!
    
    # Audio
    "librosa==0.10.2",
    "soundfile==0.12.1",
    "pyworld==0.3.4",
    
    # ModelScope & Dependencies
    "modelscope==1.20.0",
    "HyperPyYAML==1.2.2",
    "conformer==0.3.2",
    "wetext==0.0.4",
    "x-transformers==2.11.24",
    "diffusers==0.29.0",
    
    # ML Tools
    "lightning==2.2.4",
    "onnx==1.16.0",
    "onnxruntime-gpu==1.18.0; sys_platform == 'linux'",
    "onnxruntime==1.18.0; sys_platform == 'darwin' or sys_platform == 'win32'",
    
    # Utilities
    "hydra-core==1.3.2",
    "omegaconf==2.3.0",
    "protobuf==4.25.0",
    "pyarrow==18.1.0",
    "networkx==3.1.0",
    "inflect==7.3.1",
    "gdown==5.1.0",
    "wget==3.2.0",
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
        "gptsovits": GPTSOVITS_DEPS,  # transformers<=4.50
        "cosyvoice": COSYVOICE_DEPS,  # transformers==4.51.3
        
        # ============================================================
        # 조합 엔진 (버전 충돌 주의!)
        # ============================================================
        
        # Supertonic + GPT-SoVITS (호환 보장! ✅)
        "supertonic-gptsovits": SUPERTONIC_DEPS + GPTSOVITS_DEPS,
        
        # ⚠️ WARNING: 아래 조합들은 transformers 버전 충돌!
        # → Docker 사용 강력 권장!
        
        # Supertonic + CosyVoice (torch==2.3.1 고정)
        "supertonic-cosyvoice": SUPERTONIC_DEPS + COSYVOICE_DEPS,
        
        # CosyVoice + GPT-SoVITS (❌ transformers 충돌! Docker 필수!)
        # CosyVoice: transformers==4.51.3
        # GPT-SoVITS: transformers<=4.50
        "cosyvoice-gptsovits": COSYVOICE_DEPS + GPTSOVITS_DEPS,
        
        # 전체 설치 (❌ 충돌! Docker 강력 권장!)
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
