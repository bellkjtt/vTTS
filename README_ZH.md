# vTTS - 通用TTS/STT服务系统

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**语音领域的vLLM** - 从Huggingface直接下载和推理的通用TTS/STT服务系统

[한국어](README.md) | [English](README_EN.md) | 中文 | [日本語](README_JA.md)

## 🎯 目标

- 🚀 **简单易用**: 一行命令 `vtts serve model-name` 启动服务器
- 🤗 **Huggingface集成**: 自动下载和缓存模型
- 🌐 **OpenAI兼容API**: 完全兼容OpenAI TTS & Whisper API
- 🎙️ **TTS + STT集成**: 文本转语音和语音识别统一
- 🐳 **Docker支持**: 无依赖冲突同时运行多个引擎
- 🎮 **CUDA支持**: GPU加速快速推理

## 📦 支持的模型

### TTS (文本转语音)
| 引擎 | 速度 | 质量 | 多语言 | 语音克隆 | 参考音频 |
|------|------|------|--------|----------|---------|
| ✅ **Supertonic-2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 5种语言 | ❌ | 不需要 |
| ✅ **GPT-SoVITS v3** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 5种语言 | ✅ Zero-shot | **必需** |
| ✅ **CosyVoice3** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 9种语言 | ⚠️ | 可选 |
| 🔜 **StyleTTS2**, **XTTS-v2**, **Bark** | - | - | - | - | - |

> **GPT-SoVITS**: Zero-shot语音克隆模型。需要3-10秒参考音频。

### STT (语音转文本)
- ✅ **Faster-Whisper** - 超快速Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

---

## 🚀 快速开始

### 方法1: 仅Supertonic (最简单)

```bash
# 默认安装 (自动支持GPU)
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# 仅CPU (无GPU环境)
pip install "vtts[supertonic-cpu] @ git+https://github.com/bellkjtt/vTTS.git"

# 启动服务器
vtts serve Supertone/supertonic-2 --device cuda
```

### 方法2: GPT-SoVITS安装 (语音克隆)

```bash
# 1. 安装vTTS基础
pip install git+https://github.com/bellkjtt/vTTS.git

# 2. 自动安装GPT-SoVITS (自动克隆仓库 + 安装依赖!)
vtts setup --engine gptsovits

# 3. 启动服务器
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

> 💡 `vtts setup` 自动将GPT-SoVITS克隆到 `~/.vtts/GPT-SoVITS` 并安装依赖。

### 方法3: Docker (推荐用于多引擎)

```bash
# Supertonic (最快)
docker-compose up -d supertonic   # :8001

# GPT-SoVITS (语音克隆) - 需要reference_audio卷
mkdir -p reference_audio
docker-compose up -d gptsovits    # :8002

# CosyVoice (高质量)
docker-compose up -d cosyvoice    # :8003

# 全部 + API网关
docker-compose --profile gateway up -d  # :8000
```

📖 详情: [Docker指南](DOCKER.md)

### 方法4: CLI自动安装

```bash
# 安装基础，然后添加引擎
pip install git+https://github.com/bellkjtt/vTTS.git

vtts setup --engine supertonic --cuda   # Supertonic + CUDA
vtts setup --engine gptsovits           # GPT-SoVITS (包括仓库克隆)
vtts setup --engine all                 # 所有引擎
```

---

## 🔧 环境设置

### 诊断和自动修复

```bash
# 诊断环境
vtts doctor

# 自动修复 (numpy, onnxruntime兼容性)
vtts doctor --fix

# 强制安装CUDA
vtts doctor --fix --cuda
```

示例输出:
```
🩺 vTTS环境诊断

✓ Python: 3.10.12
✓ numpy: 1.26.4
✓ onnxruntime: 1.16.0 (支持CUDA)
  Providers: CUDAExecutionProvider, CPUExecutionProvider
✓ PyTorch: 2.1.0 (CUDA 12.1)
  GPU: NVIDIA GeForce RTX 4090
✓ vTTS: 已安装

✅ 所有环境都已就绪!
```

### 在Kaggle/Colab上

```python
# 安装 + 自动配置
!pip install -q git+https://github.com/bellkjtt/vTTS.git
!vtts doctor --fix --cuda
```

---

## 💻 启动服务器

### Supertonic (快速TTS)
```bash
vtts serve Supertone/supertonic-2
vtts serve Supertone/supertonic-2 --device cuda --port 8000
```

### GPT-SoVITS (语音克隆)
```bash
# 需要克隆GPT-SoVITS仓库! (参见上面的"方法2")
# 检查环境变量
echo $GPT_SOVITS_PATH  # 应输出 ~/.vtts/GPT-SoVITS

# 启动服务器
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

### TTS + STT同时
```bash
vtts serve Supertone/supertonic-2 --stt-model large-v3
vtts serve Supertone/supertonic-2 --stt-model base --device cuda
```

### 可用选项
| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 服务器主机 |
| `--port` | 8000 | 服务器端口 |
| `--device` | auto | cuda, cpu, auto |
| `--stt-model` | None | Whisper模型 (base, large-v3等) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

---

## 🐍 Python使用

### 基本用法
```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")

# TTS
audio = client.tts(
    text="你好，这是vTTS。",
    voice="F1",
    language="zh",
    speed=1.05
)
audio.save("output.wav")

# STT
text = client.stt("audio.wav")
print(text)
```

### 高级选项 (Supertonic)
```python
audio = client.tts(
    text="你好世界",
    voice="F1",           # M1-M4, F1-F4
    language="zh",        # en, ko, es, pt, fr, zh
    speed=1.05,           # 速度 (默认: 1.05)
    total_steps=5,        # 质量 (1-20, 默认: 5)
    silence_duration=0.3  # 块之间的静音 (秒)
)
```

### 语音克隆 (GPT-SoVITS)
```python
from vtts import VTTSClient

# GPT-SoVITS客户端 (需要参考音频!)
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="这是语音克隆测试。",
    model="kevinwang676/GPT-SoVITS-v3",
    voice="reference",
    language="zh",
    reference_audio="./samples/reference.wav",  # 参考音频 (必需!)
    reference_text="参考音频说的内容"  # 参考文本 (必需!)
)
audio.save("cloned_voice.wav")
```
> ⚠️ GPT-SoVITS需要 `reference_audio` 和 `reference_text` 参数!

### OpenAI SDK兼容
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="你好，很高兴见到你。"
)
response.stream_to_file("output.mp3")
```

### cURL
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好!", "voice": "F1", "model": "Supertone/supertonic-2"}' \
  --output output.mp3
```

---

## 🐳 Docker

### 端口配置
| 引擎 | 端口 | GPU内存 |
|------|------|---------|
| Gateway (Nginx) | 8000 | - |
| Supertonic | 8001 | ~1GB |
| GPT-SoVITS | 8002 | ~4GB |
| CosyVoice | 8003 | ~3GB |

### 快速开始
```bash
# 构建镜像
docker-compose build

# 运行
docker-compose up -d supertonic   # 仅Supertonic
docker-compose up -d              # 全部

# 日志
docker-compose logs -f supertonic

# 停止
docker-compose down
```

📖 详情: [Docker指南](DOCKER.md)

---

## 📊 CLI命令

| 命令 | 说明 |
|------|------|
| `vtts serve MODEL` | 启动TTS服务器 |
| `vtts doctor` | 诊断环境 |
| `vtts doctor --fix` | 自动修复环境 |
| `vtts setup --engine ENGINE` | 按引擎安装 |
| `vtts list-models` | 列出支持的模型 |
| `vtts info MODEL` | 模型信息 |

---

## 🏗️ 架构

```
vTTS/
├── vtts/
│   ├── __init__.py           # 自动环境检查
│   ├── cli.py                # CLI (serve, doctor, setup)
│   ├── client.py             # Python客户端
│   ├── server/
│   │   ├── app.py            # FastAPI应用
│   │   ├── routes.py         # TTS API路由
│   │   ├── stt_routes.py     # STT API路由
│   │   └── models.py         # Pydantic模型
│   ├── engines/
│   │   ├── base.py           # 基础引擎接口
│   │   ├── registry.py       # 自动引擎注册
│   │   ├── supertonic.py     # Supertonic引擎
│   │   ├── gptsovits.py      # GPT-SoVITS引擎
│   │   ├── cosyvoice.py      # CosyVoice引擎
│   │   └── _supertonic/      # 嵌入式ONNX模块
│   └── utils/
│       └── audio.py          # 音频处理
├── docker/
│   ├── Dockerfile.supertonic
│   ├── Dockerfile.gptsovits
│   ├── Dockerfile.cosyvoice
│   └── nginx.conf            # API网关
├── docker-compose.yml
├── setup.py
└── README.md
```

---

## 🔧 开发路线图

- [x] 项目结构设计
- [x] 基础引擎接口
- [x] Supertonic-2引擎
- [x] CosyVoice3引擎
- [x] GPT-SoVITS引擎
- [x] FastAPI服务器
- [x] OpenAI兼容API
- [x] CLI实现 (serve, doctor, setup)
- [x] 自动模型下载
- [x] CUDA支持
- [x] Docker镜像
- [x] 自动环境诊断/修复
- [ ] 流式支持
- [ ] 批量推理优化

---

## 📚 文档

- [快速开始指南](QUICKSTART.md)
- [故障排除指南](TROUBLESHOOTING.md)
- [Docker指南](DOCKER.md)
- [Kaggle测试笔记本](kaggle_test_notebook.ipynb)
- [示例代码](examples/)

---

## ⚠️ 故障排除

### numpy兼容性错误
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**解决方案**: `vtts doctor --fix`

### 找不到CUDA
```
WARNING: CUDA requested but CUDAExecutionProvider not available
```
**解决方案**: `vtts doctor --fix --cuda`

### 依赖冲突
**解决方案**: 使用Docker
```bash
docker-compose up -d supertonic
```

📖 更多问题: [故障排除指南](TROUBLESHOOTING.md)

---

## 📝 许可证

Apache许可证 2.0

## 💖 赞助

这个项目对你有帮助吗?

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 架构灵感
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
