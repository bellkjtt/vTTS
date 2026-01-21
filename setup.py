"""Setup script for vTTS"""

from setuptools import setup, find_packages
from pathlib import Path

# README 읽기
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

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
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "click>=8.1.0",
        "httpx>=0.26.0",
        "huggingface-hub>=0.20.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "faster-whisper>=1.0.0",
        "python-multipart>=0.0.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "supertonic": [
            "supertonic>=0.1.0",
        ],
        "cosyvoice": [
            "modelscope>=1.9.0",
            "HyperPyYAML>=1.2.0",
            "conformer>=0.3.0",
            "wetext>=0.0.4",
            "x-transformers>=2.11.0",
            "diffusers>=0.29.0",
        ],
        "gptsovits": [
            "funasr>=1.0.27",
            "peft>=0.10.0,<0.18.0",
            "g2p_en",
            "pyopenjtalk>=0.4.1",
            "cn2an",
            "pypinyin",
            "sentencepiece",
            "jieba",
            "fast_langdetect>=0.3.1",
            "g2pk2",
            "ko_pron",
        ],
        "all": [
            "supertonic>=0.1.0",
            "modelscope>=1.9.0",
            "HyperPyYAML>=1.2.0",
            "conformer>=0.3.0",
            "wetext>=0.0.4",
            "x-transformers>=2.11.0",
            "diffusers>=0.29.0",
            "funasr>=1.0.27",
            "peft>=0.10.0,<0.18.0",
            "g2p_en",
            "pyopenjtalk>=0.4.1",
            "cn2an",
            "pypinyin",
            "sentencepiece",
            "jieba",
            "fast_langdetect>=0.3.1",
            "g2pk2",
            "ko_pron",
        ],
    },
    entry_points={
        "console_scripts": [
            "vtts=vtts.cli:main",
        ],
    },
)
