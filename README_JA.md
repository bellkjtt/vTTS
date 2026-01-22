# vTTS - ユニバーサルTTS/STTサービングシステム

[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/bellkjtt/vTTS/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/bellkjtt)](https://github.com/sponsors/bellkjtt)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/bellkjtt/vTTS)

**音声のためのvLLM** - Huggingfaceから直接ダウンロードして推論可能なユニバーサルTTS/STTサービングシステム

[한국어](README.md) | [English](README_EN.md) | [中文](README_ZH.md) | 日本語

## 🎯 目標

- 🚀 **シンプルな使い方**: `vtts serve model-name` 一行でサーバー起動
- 🤗 **Huggingface統合**: モデルの自動ダウンロードとキャッシング
- 🌐 **OpenAI互換API**: OpenAI TTS & Whisper APIと完全互換
- 🎙️ **TTS + STT統合**: テキスト音声変換と音声認識の統合
- 🐳 **Docker対応**: 依存関係の競合なしで複数のエンジンを同時実行
- 🎮 **CUDA対応**: GPUアクセラレーションによる高速推論

## 📦 対応モデル

### TTS (Text-to-Speech)
| エンジン | 速度 | 品質 | 多言語 | 音声クローン | 参照音声 |
|---------|------|------|--------|-------------|---------|
| ✅ **Supertonic-2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 5言語 | ❌ | 不要 |
| ✅ **GPT-SoVITS v3** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 5言語 | ✅ Zero-shot | **必須** |
| ✅ **CosyVoice3** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 9言語 | ⚠️ | オプション |
| 🔜 **StyleTTS2**, **XTTS-v2**, **Bark** | - | - | - | - | - |

> **GPT-SoVITS**: Zero-shot音声クローンモデル。3-10秒の参照音声が必要です。

### STT (Speech-to-Text)
- ✅ **Faster-Whisper** - 超高速Whisper (CTranslate2)
- 🔜 **Whisper.cpp**, **Parakeet**

---

## 🚀 クイックスタート

### 方法1: Supertonicのみ使用 (最もシンプル)

```bash
# デフォルトインストール (GPU自動対応)
pip install "vtts[supertonic] @ git+https://github.com/bellkjtt/vTTS.git"

# CPUのみ (GPU無し環境)
pip install "vtts[supertonic-cpu] @ git+https://github.com/bellkjtt/vTTS.git"

# サーバー起動
vtts serve Supertone/supertonic-2 --device cuda
```

### 方法2: GPT-SoVITSセットアップ (音声クローン)

```bash
# 1. vTTSベースインストール
pip install git+https://github.com/bellkjtt/vTTS.git

# 2. GPT-SoVITS自動インストール (リポジトリクローン + 依存関係自動処理!)
vtts setup --engine gptsovits

# 3. サーバー起動
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

> 💡 `vtts setup` は GPT-SoVITSを `~/.vtts/GPT-SoVITS` に自動クローンし、依存関係をインストールします。

### 方法3: Docker (複数エンジン推奨)

```bash
# Supertonic (最速)
docker-compose up -d supertonic   # :8001

# GPT-SoVITS (音声クローン) - reference_audioボリューム必要
mkdir -p reference_audio
docker-compose up -d gptsovits    # :8002

# CosyVoice (高品質)
docker-compose up -d cosyvoice    # :8003

# 全て + APIゲートウェイ
docker-compose --profile gateway up -d  # :8000
```

📖 詳細: [Dockerガイド](DOCKER.md)

### 方法4: CLI自動インストール

```bash
# ベースインストール後、エンジン追加
pip install git+https://github.com/bellkjtt/vTTS.git

vtts setup --engine supertonic --cuda   # Supertonic + CUDA
vtts setup --engine gptsovits           # GPT-SoVITS (リポジトリクローン含む)
vtts setup --engine all                 # 全エンジン
```

---

## 🔧 環境設定

### 診断と自動修復

```bash
# 環境診断
vtts doctor

# 自動修復 (numpy, onnxruntime互換性)
vtts doctor --fix

# CUDA強制インストール
vtts doctor --fix --cuda
```

出力例:
```
🩺 vTTS環境診断

✓ Python: 3.10.12
✓ numpy: 1.26.4
✓ onnxruntime: 1.16.0 (CUDA対応)
  Providers: CUDAExecutionProvider, CPUExecutionProvider
✓ PyTorch: 2.1.0 (CUDA 12.1)
  GPU: NVIDIA GeForce RTX 4090
✓ vTTS: インストール済み

✅ すべての環境が整っています!
```

### Kaggle/Colabで

```python
# インストール + 自動設定
!pip install -q git+https://github.com/bellkjtt/vTTS.git
!vtts doctor --fix --cuda
```

---

## 💻 サーバー起動

### Supertonic (高速TTS)
```bash
vtts serve Supertone/supertonic-2
vtts serve Supertone/supertonic-2 --device cuda --port 8000
```

### GPT-SoVITS (音声クローン)
```bash
# GPT-SoVITSリポジトリクローン必要! (上記「方法2」参照)
# 環境変数確認
echo $GPT_SOVITS_PATH  # ~/.vtts/GPT-SoVITS が出力されるべき

# サーバー起動
vtts serve kevinwang676/GPT-SoVITS-v3 --device cuda --port 8002
```

### TTS + STT同時
```bash
vtts serve Supertone/supertonic-2 --stt-model large-v3
vtts serve Supertone/supertonic-2 --stt-model base --device cuda
```

### 利用可能なオプション
| オプション | デフォルト | 説明 |
|----------|---------|------|
| `--host` | 0.0.0.0 | サーバーホスト |
| `--port` | 8000 | サーバーポート |
| `--device` | auto | cuda, cpu, auto |
| `--stt-model` | None | Whisperモデル (base, large-v3等) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

---

## 🐍 Python使用法

### 基本的な使い方
```python
from vtts import VTTSClient

client = VTTSClient("http://localhost:8000")

# TTS
audio = client.tts(
    text="こんにちは、vTTSです。",
    voice="F1",
    language="ja",
    speed=1.05
)
audio.save("output.wav")

# STT
text = client.stt("audio.wav")
print(text)
```

### 高度なオプション (Supertonic)
```python
audio = client.tts(
    text="こんにちは世界",
    voice="F1",           # M1-M4, F1-F4
    language="ja",        # en, ko, es, pt, fr, ja
    speed=1.05,           # 速度 (デフォルト: 1.05)
    total_steps=5,        # 品質 (1-20, デフォルト: 5)
    silence_duration=0.3  # チャンク間の無音 (秒)
)
```

### 音声クローン (GPT-SoVITS)
```python
from vtts import VTTSClient

# GPT-SoVITSクライアント (参照音声必須!)
client = VTTSClient("http://localhost:8002")

audio = client.tts(
    text="これは音声クローンのテストです。",
    model="kevinwang676/GPT-SoVITS-v3",
    voice="reference",
    language="ja",
    reference_audio="./samples/reference.wav",  # 参照音声 (必須!)
    reference_text="参照音声で話している内容",  # 参照テキスト (必須!)
    # 🎛️ 品質調整パラメータ (オプション)
    speed=1.0,                  # 速度 (0.5-2.0)
    top_k=15,                   # Top-Kサンプリング (1-100)
    top_p=1.0,                  # Top-Pサンプリング (0.0-1.0)
    temperature=1.0,            # 多様性 (0.1-2.0, 低いほど安定)
    sample_steps=32,            # サンプリングステップ (1-100, 高いほど高品質)
    seed=-1,                    # ランダムシード (-1: ランダム, 固定値: 再現可能)
    repetition_penalty=1.35,    # 繰り返しペナルティ (1.0-2.0, 高いほど繰り返し減少)
    text_split_method="cut5",   # テキスト分割方法 (cut5, four_sentences等)
    batch_size=1,               # バッチサイズ (1-10)
    fragment_interval=0.3,      # フラグメント間隔秒 (0.0-2.0)
    parallel_infer=True         # 並列推論を有効化
)
audio.save("cloned_voice.wav")
```
> ⚠️ GPT-SoVITSは `reference_audio` と `reference_text` パラメータが必須です!

**パラメータガイド:**
| パラメータ | デフォルト | 範囲 | 説明 |
|---------|-------|------|------|
| `top_k` | 15 | 1-100 | Top-Kサンプリング (低いほど保守的) |
| `top_p` | 1.0 | 0.0-1.0 | Nucleusサンプリング (低いほど集中的) |
| `temperature` | 1.0 | 0.1-2.0 | 生成の多様性 (低いほど安定) |
| `sample_steps` | 32 | 1-100 | サンプリングステップ (高いほど高品質) |
| `seed` | -1 | -1または正数 | ランダムシード (-1: ランダム) |
| `repetition_penalty` | 1.35 | 1.0-2.0 | 繰り返しペナルティ (高いほど繰り返し減少) |
| `text_split_method` | cut5 | - | テキスト分割方法 |
| `batch_size` | 1 | 1-10 | バッチサイズ |
| `fragment_interval` | 0.3 | 0.0-2.0 | フラグメント間の無音 (秒) |
| `parallel_infer` | True | bool | 並列推論 |

**シナリオ別推奨:**
- **高品質/安定**: `temperature=0.7, top_p=0.9, sample_steps=40, repetition_penalty=1.5`
- **高速生成**: `sample_steps=16, top_k=10, batch_size=2`
- **多様な結果**: `temperature=1.2, top_k=30, repetition_penalty=1.2`
- **長文**: `text_split_method="four_sentences", fragment_interval=0.5`

### OpenAI SDK互換
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.audio.speech.create(
    model="Supertone/supertonic-2",
    voice="F1",
    input="こんにちは、お会いできて嬉しいです。"
)
response.stream_to_file("output.mp3")
```

### cURL
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "こんにちは!", "voice": "F1", "model": "Supertone/supertonic-2"}' \
  --output output.mp3
```

---

## 🐳 Docker

### ポート構成
| エンジン | ポート | GPUメモリ |
|---------|------|----------|
| Gateway (Nginx) | 8000 | - |
| Supertonic | 8001 | ~1GB |
| GPT-SoVITS | 8002 | ~4GB |
| CosyVoice | 8003 | ~3GB |

### クイックスタート
```bash
# イメージビルド
docker-compose build

# 実行
docker-compose up -d supertonic   # Supertonicのみ
docker-compose up -d              # 全て

# ログ
docker-compose logs -f supertonic

# 停止
docker-compose down
```

📖 詳細: [Dockerガイド](DOCKER.md)

---

## 📊 CLIコマンド

| コマンド | 説明 |
|---------|------|
| `vtts serve MODEL` | TTSサーバー起動 |
| `vtts doctor` | 環境診断 |
| `vtts doctor --fix` | 環境自動修復 |
| `vtts setup --engine ENGINE` | エンジン別インストール |
| `vtts list-models` | 対応モデル一覧 |
| `vtts info MODEL` | モデル情報 |

---

## 🏗️ アーキテクチャ

```
vTTS/
├── vtts/
│   ├── __init__.py           # 自動環境チェック
│   ├── cli.py                # CLI (serve, doctor, setup)
│   ├── client.py             # Pythonクライアント
│   ├── server/
│   │   ├── app.py            # FastAPIアプリ
│   │   ├── routes.py         # TTS APIルート
│   │   ├── stt_routes.py     # STT APIルート
│   │   └── models.py         # Pydanticモデル
│   ├── engines/
│   │   ├── base.py           # ベースエンジンインターフェース
│   │   ├── registry.py       # 自動エンジン登録
│   │   ├── supertonic.py     # Supertonicエンジン
│   │   ├── gptsovits.py      # GPT-SoVITSエンジン
│   │   ├── cosyvoice.py      # CosyVoiceエンジン
│   │   └── _supertonic/      # 組み込みONNXモジュール
│   └── utils/
│       └── audio.py          # オーディオ処理
├── docker/
│   ├── Dockerfile.supertonic
│   ├── Dockerfile.gptsovits
│   ├── Dockerfile.cosyvoice
│   └── nginx.conf            # APIゲートウェイ
├── docker-compose.yml
├── setup.py
└── README.md
```

---

## 🔧 開発ロードマップ

- [x] プロジェクト構造設計
- [x] ベースエンジンインターフェース
- [x] Supertonic-2エンジン
- [x] CosyVoice3エンジン
- [x] GPT-SoVITSエンジン
- [x] FastAPIサーバー
- [x] OpenAI互換API
- [x] CLI実装 (serve, doctor, setup)
- [x] 自動モデルダウンロード
- [x] CUDA対応
- [x] Dockerイメージ
- [x] 自動環境診断/修復
- [ ] ストリーミング対応
- [ ] バッチ推論最適化

---

## 📚 ドキュメント

- [クイックスタートガイド](QUICKSTART.md)
- [トラブルシューティングガイド](TROUBLESHOOTING.md)
- [Dockerガイド](DOCKER.md)
- [Kaggleテストノートブック](kaggle_test_notebook.ipynb)
- [サンプルコード](examples/)

---

## ⚠️ トラブルシューティング

### numpy互換性エラー
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**解決方法**: `vtts doctor --fix`

### CUDAが見つからない
```
WARNING: CUDA requested but CUDAExecutionProvider not available
```
**解決方法**: `vtts doctor --fix --cuda`

### 依存関係の競合
**解決方法**: Dockerを使用
```bash
docker-compose up -d supertonic
```

📖 その他の問題: [トラブルシューティングガイド](TROUBLESHOOTING.md)

---

## 📝 ライセンス

Apacheライセンス 2.0

## 💖 スポンサー

このプロジェクトは役に立ちましたか?

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge)](https://github.com/sponsors/bellkjtt)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee-orange?style=for-the-badge)](https://ko-fi.com/bellkjtt)

## 🙏 謝辞

- [vLLM](https://github.com/vllm-project/vllm) - アーキテクチャのインスピレーション
- [Supertone](https://huggingface.co/Supertone/supertonic-2)
- [FunAudioLLM](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [GPT-SoVITS](https://huggingface.co/kevinwang676/GPT-SoVITS-v3)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
