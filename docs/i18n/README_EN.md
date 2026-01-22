# vTTS - Universal TTS/STT Serving System

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**vLLM for Speech** - Universal TTS/STT serving system with direct Huggingface download and inference

[한국어](../../README.md) | English | [中文](README_ZH.md) | [日本語](README_JA.md)

## Goals

- **Simple Usage**: Start server with one line `vtts serve model-name`
- **Huggingface Integration**: Automatic model download and caching
- **OpenAI Compatible API**: Fully compatible with OpenAI TTS & Whisper API
- **TTS + STT Integration**: Text-to-Speech and Speech-to-Text unified
- **Docker Support**: Run multiple engines simultaneously without dependency conflicts
- **CUDA Support**: Fast inference with GPU acceleration

## Supported Models

### TTS (Text-to-Speech)
| Engine | Speed | Quality | Multilingual | Voice Cloning | Reference Audio |
|--------|-------|---------|--------------|---------------|-----------------|
| **Supertonic-2** | Very Fast | Good | 5 languages | No | Not required |
| **GPT-SoVITS v3** | Moderate | Excellent | 5 languages | Zero-shot | **Required** |
| **CosyVoice3** | Fast | Very Good | 9 languages | Optional | Optional |
| **StyleTTS2**, **XTTS-v2**, **Bark** (Coming Soon) | - | - | - | - | - |

> **GPT-SoVITS**: Zero-shot voice cloning model. Requires 3-10 second reference audio.

### STT (Speech-to-Text)
- **Faster-Whisper** - Ultra-fast Whisper (CTranslate2)
- **Whisper.cpp**, **Parakeet** (Coming Soon)

---

## Quick Start

> **NOTE - Dependency Conflicts**  
> Each engine has different dependencies. **For local installation, install only one engine at a time.**  
> To use multiple engines simultaneously, **using Docker is strongly recommended!**

### Local Installation

#### Option 1: Supertonic Only (Simplest)

```bash
# Default install (GPU auto-support)
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# Start server
vtts serve Supertone/supertonic-2 --device cuda
```

#### Option 2: Supertonic + GPT-SoVITS (Compatible!)

```bash
# 1. Combined installation (dependencies verified)
pip install "vtts[supertonic-gptsovits] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. Auto-clone GPT-SoVITS repository
vtts setup --engine gptsovits

# 3. Start servers (different ports)
vtts serve Supertone/supertonic-2 --port 8001 --device cuda
vtts serve kevinwang676/GPT-SoVITS-v3 --port 8002 --device cuda
```

> **Supertonic + GPT-SoVITS can be installed together without conflicts!**

#### Option 3: CosyVoice Only (Separate Environment Recommended)

```bash
# 1. Install
pip install "vtts[cosyvoice] @ git+https://github.com/bellkjtt/vTTS.git"

# 2. Auto-clone CosyVoice repository
vtts setup --engine cosyvoice

# 3. Start server
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --device cuda
```

> **CosyVoice may have dependency conflicts. Use separate virtual environment or Docker!**

### Docker (Multiple Engines)

```bash
# Individual services
docker-compose up -d supertonic   # :8001
docker-compose up -d gptsovits    # :8002 (reference_audio folder required)
docker-compose up -d cosyvoice    # :8003

# All + Nginx API Gateway
docker-compose --profile gateway up -d  # :8000 (unified endpoint)
```

Details: [Docker Guide](../../DOCKER.md)

### CLI Auto-Install

```bash
# Base installation
pip install git+https://github.com/bellkjtt/vTTS.git

# Engine-specific auto-install (repo clone + dependencies)
vtts setup --engine supertonic           # Supertonic only
vtts setup --engine gptsovits            # GPT-SoVITS (auto-clone)
vtts setup --engine cosyvoice            # CosyVoice (auto-clone)
```

---

## Environment Setup

### Diagnose and Auto-Fix

```bash
# Diagnose environment
vtts doctor

# Auto-fix (numpy, onnxruntime compatibility)
vtts doctor --fix

# Force CUDA installation
vtts doctor --fix --cuda
```

Example output:
```
vTTS Environment Diagnosis

✓ Python: 3.10.12
✓ numpy: 1.26.4
✓ onnxruntime: 1.16.0 (CUDA supported)
  Providers: CUDAExecutionProvider, CPUExecutionProvider
✓ PyTorch: 2.1.0 (CUDA 12.1)
  GPU: NVIDIA GeForce RTX 4090
✓ vTTS: Installed

All environments are ready!
```

### On Kaggle/Colab

```python
# Install + auto-configure
!pip install -q git+https://github.com/bellkjtt/vTTS.git
!vtts doctor --fix --cuda
```

---

## Starting Server

### Supertonic (Fast TTS)
```bash
vtts serve Supertone/supertonic-2
vtts serve Supertone/supertonic-2 --device cuda --port 8000
```

### GPT-SoVITS (Voice Cloning)

```bash
# Install GPT-SoVITS repository (see "Option 2" above)
vtts setup --engine gptsovits

# Start server (pretrained models auto-downloaded!)
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

**Note:**
- On first run, **automatically downloads pretrained models** from [HuggingFace](https://huggingface.co/kevinwang676/GPT-SoVITS-v3/tree/main/GPT_SoVITS/pretrained_models) (~2.9 GB)
- Models are cached in `~/.cache/huggingface/` and reused later

### TTS + STT Simultaneous
```bash
vtts serve Supertone/supertonic-2 --stt-model large-v3
vtts serve Supertone/supertonic-2 --stt-model base --device cuda
```

### Available Options
| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Server host |
| `--port` | 8000 | Server port |
| `--device` | auto | cuda, cpu, auto |
| `--stt-model` | None | Whisper model (base, large-v3, etc) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

---

## Python Usage

### Basic Usage
```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")

# TTS
audio = client.tts(
    text="Hello, this is vTTS.",
    voice="F1",
    language="en",
    speed=1.05
)
audio.save("output.wav")

# STT
text = client.stt("audio.wav")
print(text)
```

### Advanced Options (Supertonic)
```python
audio = client.tts(
    text="Hello world",
    voice="F1",           # M1-M4, F1-F4
    language="en",        # en, ko, es, pt, fr
    speed=1.05,           # Speed (default: 1.05)
    total_steps=5,        # Quality (1-20, default: 5)
    silence_duration=0.3  # Silence between chunks (seconds)
)
```

### Voice Cloning (GPT-SoVITS)
```python
from vtts import VTTSClient

# GPT-SoVITS client (reference audio required!)
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="This is a voice cloning test.",
    model="kevinwang676/GPT-SoVITS-v3",
    voice="reference",
    language="en",
    reference_audio="./samples/reference.wav",  # Reference audio (required!)
    reference_text="This is what the reference audio says",  # Reference text (required!)
    # Quality Control Parameters (optional)
    speed=1.0,                  # Speed (0.5-2.0)
    top_k=15,                   # Top-K sampling (1-100)
    top_p=1.0,                  # Top-P sampling (0.0-1.0)
    temperature=1.0,            # Diversity (0.1-2.0, lower = more stable)
    sample_steps=32,            # Sampling steps (1-100, higher = better quality)
    seed=-1,                    # Random seed (-1: random, fixed: reproducible)
    repetition_penalty=1.35,    # Repetition penalty (1.0-2.0, higher = less repetition)
    text_split_method="cut5",   # Text splitting method (cut5, four_sentences, etc)
    batch_size=1,               # Batch size (1-10)
    fragment_interval=0.3,      # Fragment interval in seconds (0.0-2.0)
    parallel_infer=True         # Enable parallel inference
)
audio.save("cloned_voice.wav")
```
> **NOTE**: GPT-SoVITS requires `reference_audio` and `reference_text` parameters!

**Parameter Guide:**
| Parameter | Default | Range | Description |
|---------|-------|------|------|
| `top_k` | 15 | 1-100 | Top-K sampling (lower = more conservative) |
| `top_p` | 1.0 | 0.0-1.0 | Nucleus sampling (lower = more focused) |
| `temperature` | 1.0 | 0.1-2.0 | Generation diversity (lower = more stable) |
| `sample_steps` | 32 | 1-100 | Sampling steps (higher = better quality) |
| `seed` | -1 | -1 or positive | Random seed (-1: random) |
| `repetition_penalty` | 1.35 | 1.0-2.0 | Repetition penalty (higher = less repetition) |
| `text_split_method` | cut5 | - | Text splitting method |
| `batch_size` | 1 | 1-10 | Batch size |
| `fragment_interval` | 0.3 | 0.0-2.0 | Silence between fragments (seconds) |
| `parallel_infer` | True | bool | Parallel inference |

**Scenario Recommendations:**
- **High Quality/Stable**: `temperature=0.7, top_p=0.9, sample_steps=40, repetition_penalty=1.5`
- **Fast Generation**: `sample_steps=16, top_k=10, batch_size=2`
- **Diverse Results**: `temperature=1.2, top_k=30, repetition_penalty=1.2`
- **Long Text**: `text_split_method="four_sentences", fragment_interval=0.5`

### OpenAI SDK Compatible
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="Hello, nice to meet you."
)
response.stream_to_file("output.mp3")
```

### cURL
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "F1", "model": "Supertone/supertonic-2"}' \
  --output output.mp3
```

---

## Docker

### Port Configuration
| Engine | Port | GPU Memory |
|--------|------|-----------|
| Gateway (Nginx) | 8000 | - |
| Supertonic | 8001 | ~1GB |
| GPT-SoVITS | 8002 | ~4GB |
| CosyVoice | 8003 | ~3GB |

### Quick Start
```bash
# Build images
docker-compose build

# Run
docker-compose up -d supertonic   # Supertonic only
docker-compose up -d              # All

# Logs
docker-compose logs -f supertonic

# Stop
docker-compose down
```

Details: [Docker Guide](../../DOCKER.md)

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `vtts serve MODEL` | Start TTS server |
| `vtts doctor` | Diagnose environment |
| `vtts doctor --fix` | Auto-fix environment |
| `vtts setup --engine ENGINE` | Install by engine |
| `vtts list-models` | List supported models |
| `vtts info MODEL` | Model information |

---

## Architecture

```
vTTS/
├── vtts/
│   ├── __init__.py           # Auto environment check
│   ├── cli.py                # CLI (serve, doctor, setup)
│   ├── client.py             # Python client
│   ├── server/
│   │   ├── app.py            # FastAPI app
│   │   ├── routes.py         # TTS API routes
│   │   ├── stt_routes.py     # STT API routes
│   │   └── models.py         # Pydantic models
│   ├── engines/
│   │   ├── base.py           # Base engine interface
│   │   ├── registry.py       # Auto engine registration
│   │   ├── supertonic.py     # Supertonic engine
│   │   ├── gptsovits.py      # GPT-SoVITS engine
│   │   ├── cosyvoice.py      # CosyVoice engine
│   │   └── _supertonic/      # Embedded ONNX module
│   └── utils/
│       └── audio.py          # Audio processing
├── docker/
│   ├── Dockerfile.supertonic
│   ├── Dockerfile.gptsovits
│   ├── Dockerfile.cosyvoice
│   └── nginx.conf            # API Gateway
├── docker-compose.yml
├── setup.py
└── README.md
```

---

## Development Roadmap

- [x] Project structure design
- [x] Base engine interface
- [x] Supertonic-2 engine
- [x] CosyVoice3 engine
- [x] GPT-SoVITS engine
- [x] FastAPI server
- [x] OpenAI compatible API
- [x] CLI implementation (serve, doctor, setup)
- [x] Automatic model download
- [x] CUDA support
- [x] Docker images
- [x] Auto environment diagnosis/fix
- [ ] Streaming support
- [ ] Batch inference optimization

---

## Documentation

### Getting Started
- [Quick Start Guide](../../docs/QUICKSTART.md)
- [Installation Guide](../../docs/INSTALL.md)
- [Engine Setup Guide](../../docs/ENGINES_SETUP.md)
- [Troubleshooting Guide](../../TROUBLESHOOTING.md)
- [Docker Guide](../../DOCKER.md)

### Examples & Tests
- [Example Code](../../examples/) - [Example README](../../examples/README.md)
- [Test Suite](../../tests/) - [Test README](../../tests/README.md)

### Developer Documentation
- [Developer Docs](../../docs/) - [Docs README](../../docs/README.md)

---

## Troubleshooting

### numpy Compatibility Error
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**Solution**: `vtts doctor --fix`

### CUDA Not Found
```
WARNING: CUDA requested but CUDAExecutionProvider not available
```
**Solution**: `vtts doctor --fix --cuda`

### Dependency Conflicts
**Solution**: Use Docker
```bash
docker-compose up -d supertonic
```

More issues: [Troubleshooting Guide](../../TROUBLESHOOTING.md)

---

## License

Apache License 2.0

## Sponsorship

Did this project help you?

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - Architecture inspiration
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
