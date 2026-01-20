# vTTS - 通用 TTS/STT 服务系统

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**语音领域的 vLLM** - 从 Huggingface 自动下载模型的通用 TTS/STT 服务系统

[한국어](README.md) | [English](README_EN.md) | 中文 | [日本語](README_JA.md)

## 🎯 目标

- 🚀 **简单易用**: 一行命令启动服务器 `vtts serve model-name`
- 🤗 **Huggingface 集成**: 自动下载和缓存模型
- 🌐 **OpenAI 兼容**: 完全兼容 OpenAI TTS 和 Whisper API
- 🎙️ **TTS + STT 集成**: 同时支持文本转语音和语音识别
- 🇰🇷 **韩语优先**: 专注于支持韩语的模型
- 🔌 **插件架构**: 轻松添加新引擎

## 📦 支持的模型

### TTS (文本转语音)
- ✅ **GPT-SoVITS-v3** - Few-shot 声音克隆
- ✅ **Supertonic-2** - 超快速设备端 TTS (5种语言)
- ✅ **CosyVoice3** - Zero-shot 多语言 TTS (9种语言，18+种中国方言)
- 🔜 **StyleTTS2**, **XTTS-v2**, **Bark**

### STT (语音转文本)
- ✅ **Faster-Whisper** - 高性能 Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

## 🚀 快速开始

### 安装

#### 从 GitHub 安装 (当前)
```bash
pip install git+https://github.com/bellkjtt/vTTS.git
```

#### 从 PyPI 安装 (即将推出)
```bash
pip install vtts
```

#### 在 Kaggle 上测试
参见 [Kaggle 笔记本](kaggle_test_notebook.ipynb)

### 启动服务器

#### 仅 TTS
```bash
# 自动下载模型并启动服务器
vtts serve Supertone/supertonic-2

# 指定端口
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

#### TTS + STT 同时运行
```bash
# 同时服务 TTS 和 STT
vtts serve Supertone/supertonic-2 --stt-model large-v3

# 指定 GPU
vtts serve kevinwang676/GPT-SoVITS-v3 --stt-model large-v3 --device cuda:0
```

### Python 使用

```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

# 生成语音
audio = client.tts(
    text="你好，感谢使用 vTTS！",
    model="Supertone/supertonic-2",
    language="zh",
    voice="default"
)

# 保存到文件
audio.save("output.wav")
```

### OpenAI SDK 兼容
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="default",
    input="你好，很高兴见到你！"
)

response.stream_to_file("output.mp3")
```

## 🎤 STT (语音转文本) 使用

### 转录
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 转录音频
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="large-v3",
        file=audio_file,
        language="zh"
    )
    print(transcription.text)
```

### 翻译 (转为英语)
```python
# 翻译为英语
with open("chinese.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="large-v3",
        file=audio_file
    )
    print(translation.text)
```

## 🏗️ 架构

```
vTTS/
├── vtts/
│   ├── engines/          # TTS/STT 引擎
│   │   ├── base.py      # 基础接口
│   │   ├── faster_whisper.py  # Faster-Whisper STT
│   │   ├── supertonic.py      # Supertonic TTS
│   │   └── cosyvoice.py       # CosyVoice TTS
│   ├── server/           # FastAPI 服务器
│   └── utils/            # 工具
└── examples/             # 使用示例
```

## 🔧 开发路线图

- [x] 项目结构设计
- [x] 基础引擎接口
- [x] Faster-Whisper STT 引擎
- [x] FastAPI 服务器
- [x] OpenAI 兼容 API
- [x] CLI 接口
- [ ] CosyVoice3 引擎
- [ ] GPT-SoVITS 引擎
- [ ] 流式支持
- [ ] 批量推理优化

## 📝 许可证

MIT License

## 💖 支持

如果这个项目对您有帮助:

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

您的支持有助于维持这个项目！

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 架构灵感
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
