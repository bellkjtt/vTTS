# vTTS - ユニバーサル TTS/STT サービングシステム

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**音声のための vLLM** - Huggingface からモデルを自動ダウンロードする汎用 TTS/STT サービングシステム

[한국어](README.md) | [English](README_EN.md) | [中文](README_ZH.md) | 日本語

## 🎯 目標

- 🚀 **簡単な使い方**: `vtts serve model-name` 一行でサーバー起動
- 🤗 **Huggingface 統合**: モデルの自動ダウンロードとキャッシュ
- 🌐 **OpenAI 互換**: OpenAI TTS & Whisper API と完全互換
- 🎙️ **TTS + STT 統合**: テキスト音声変換と音声認識を同時サポート
- 🇰🇷 **韓国語優先**: 韓国語対応モデルに焦点
- 🔌 **プラグインアーキテクチャ**: 新しいエンジンを簡単に追加

## 📦 サポートモデル

### TTS (テキスト音声変換)
- ✅ **GPT-SoVITS-v3** - Few-shot 音声クローニング
- ✅ **Supertonic-2** - 超高速オンデバイス TTS (5言語)
- ✅ **CosyVoice3** - Zero-shot 多言語 TTS (9言語、18+ 中国方言)
- 🔜 **StyleTTS2**, **XTTS-v2**, **Bark**

### STT (音声テキスト変換)
- ✅ **Faster-Whisper** - 高性能 Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

## 🚀 クイックスタート

### インストール

#### GitHub からインストール (現在)
```bash
pip install git+https://github.com/bellkjtt/vTTS.git
```

#### PyPI からインストール (近日公開)
```bash
pip install vtts
```

#### Kaggle でテスト
[Kaggle ノートブック](kaggle_test_notebook.ipynb) を参照

### サーバー起動

#### TTS のみ
```bash
# モデルを自動ダウンロードしてサーバー起動
vtts serve Supertone/supertonic-2

# ポート指定
vtts serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

#### TTS + STT 同時
```bash
# TTS と STT を同時にサービング
vtts serve Supertone/supertonic-2 --stt-model large-v3

# GPU 指定
vtts serve kevinwang676/GPT-SoVITS-v3 --stt-model large-v3 --device cuda:0
```

### Python での使用
```python
from vtts import VTTSClient

client = VTTSClient(base_url="http://localhost:8000")

# 音声生成
audio = client.tts(
    text="こんにちは、vTTS をご利用いただきありがとうございます！",
    model="Supertone/supertonic-2",
    language="ja",
    voice="default"
)

# ファイルに保存
audio.save("output.wav")
```

### OpenAI SDK 互換
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="default",
    input="こんにちは、お会いできて嬉しいです！"
)

response.stream_to_file("output.mp3")
```

## 🎤 STT (音声テキスト変換) 使用方法

### 文字起こし
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 音声を文字起こし
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="large-v3",
        file=audio_file,
        language="ja"
    )
    print(transcription.text)
```

### 翻訳 (英語へ)
```python
# 英語に翻訳
with open("japanese.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="large-v3",
        file=audio_file
    )
    print(translation.text)
```

## 🏗️ アーキテクチャ

```
vTTS/
├── vtts/
│   ├── engines/          # TTS/STT エンジン
│   │   ├── base.py      # ベースインターフェース
│   │   ├── faster_whisper.py  # Faster-Whisper STT
│   │   ├── supertonic.py      # Supertonic TTS
│   │   └── cosyvoice.py       # CosyVoice TTS
│   ├── server/           # FastAPI サーバー
│   └── utils/            # ユーティリティ
└── examples/             # 使用例
```

## 🔧 開発ロードマップ

- [x] プロジェクト構造設計
- [x] ベースエンジンインターフェース
- [x] Faster-Whisper STT エンジン
- [x] FastAPI サーバー
- [x] OpenAI 互換 API
- [x] CLI インターフェース
- [ ] CosyVoice3 エンジン
- [ ] GPT-SoVITS エンジン
- [ ] ストリーミングサポート
- [ ] バッチ推論最適化

## 📝 ライセンス

MIT License

## 💖 サポート

このプロジェクトが役に立った場合:

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

あなたのサポートがこのプロジェクトを維持するのに役立ちます！

## 🙏 謝辞

- [vLLM](https://github.com/vllm-project/vllm) - アーキテクチャのインスピレーション
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
