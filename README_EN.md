# vTTS - Universal TTS/STT Serving System

[![GitHub release](https://img.shields.io/github/v/release/bellkjtt/vTTS)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)

**vLLM for Speech** - Universal TTS/STT serving system with automatic model download from Huggingface

[한국어](README.md) | English | [中文](README_ZH.md) | [日本語](README_JA.md)

## 🎯 Goals

- 🚀 **Simple Usage**: Start server with one command `vtts serve model-name`
- 🤗 **Huggingface Integration**: Automatic model download and caching
- 🌐 **OpenAI Compatible**: Full compatibility with OpenAI TTS & Whisper API
- 🎙️ **TTS + STT Integration**: Simultaneous text-to-speech and speech-to-text support
- 🇰🇷 **Korean First**: Focus on Korean-supporting models
- 🔌 **Plugin Architecture**: Easy to add new engines

## 📦 Supported Models

### TTS (Text-to-Speech)
- ✅ **GPT-SoVITS-v3** - Few-shot voice cloning
- ✅ **Supertonic-2** - Ultra-fast on-device TTS (5 languages)
- ✅ **CosyVoice3** - Zero-shot multilingual TTS (9 languages, 18+ Chinese dialects)
- 🔜 **StyleTTS2**, **XTTS-v2**, **Bark**

### STT (Speech-to-Text)
- ✅ **Faster-Whisper** - High-performance Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

## 🚀 Quick Start

### Installation

#### Install from GitHub (Current)
```bash
pip install git+https://github.com/bellkjtt/vTTS.git
```

#### Install from PyPI (Coming Soon)
```bash
pip install vtts
```

#### Test on Kaggle
See [Kaggle Notebook](kaggle_test_notebook.ipynb)

### Start Server

#### TTS Only
```bash
# Auto-download model and start server
vtts serve Supertone/supertonic-2

# Specify port
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

#### TTS + STT Together
```bash
# Serve both TTS and STT
vtts serve Supertone/supertonic-2 --stt-model large-v3

# Specify GPU
vtts serve kevinwang676/GPT-SoVITS-v3 --stt-model large-v3 --device cuda:0
```

### Python Usage
```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

# Generate speech
audio = client.tts(
    text="Hello, thank you for using vTTS!",
    model="Supertone/supertonic-2",
    language="en",
    voice="default"
)

# Save to file
audio.save("output.wav")
```

### OpenAI SDK Compatible
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="default",
    input="Hello, nice to meet you!"
)

response.stream_to_file("output.mp3")
```

## 🎤 STT (Speech-to-Text) Usage

### Transcription
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Transcribe audio
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="large-v3",
        file=audio_file,
        language="ko"
    )
    print(transcription.text)
```

### Translation (to English)
```python
# Translate to English
with open("korean.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="large-v3",
        file=audio_file
    )
    print(translation.text)
```

## 🏗️ Architecture

```
vTTS/
├── vtts/
│   ├── engines/          # TTS/STT engines
│   │   ├── base.py      # Base interface
│   │   ├── faster_whisper.py  # Faster-Whisper STT
│   │   ├── supertonic.py      # Supertonic TTS
│   │   └── cosyvoice.py       # CosyVoice TTS
│   ├── server/           # FastAPI server
│   └── utils/            # Utilities
└── examples/             # Usage examples
```

## 🔧 Development Roadmap

- [x] Project structure design
- [x] Base engine interface
- [x] Faster-Whisper STT engine
- [x] FastAPI server
- [x] OpenAI compatible API
- [x] CLI interface
- [ ] CosyVoice3 engine
- [ ] GPT-SoVITS engine
- [ ] Streaming support
- [ ] Batch inference optimization

## 📝 License

MIT License

## 💖 Support

If this project helps you:

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

Your support helps keep this project alive!

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - Architecture inspiration
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
