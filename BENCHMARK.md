# vTTS Benchmark Results

## Test Environment

| Item | Value |
|------|-------|
| Date | 2026-01-25T13:32:48.702996 |
| Python | 3.11.11 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 3090 Ti |

## Test Configuration

- **Runs per engine**: 5
- **Korean text**: "안녕하세요, 저는 인공지능 음성 합성 시스템입니다. 오늘 날씨가 정말 좋네요."
- **English text**: "Hello, I am an artificial intelligence speech synthesis system. The weather is really nice today."

## Summary (Sorted by RTF)

| Rank | Engine | Model | Avg Time | Avg Audio | Avg RTF |
|------|--------|-------|----------|-----------|---------|
| 1 | **Supertonic** | Supertone/supertonic-2 | 61ms | 7.80s | 0.0078 |
| 2 | **Chatterbox** | ResembleAI/chatterbox | 1405ms | 5.02s | 0.2798 |
| 3 | **KaniTTS** | nineninesix/kani-tts-370m | 2308ms | 7.50s | 0.3076 |
| 4 | **CosyVoice** | FunAudioLLM/CosyVoice2-0.5B | 8034ms | 23.20s | 0.3463 |
| 5 | **GPT-SoVITS** | kevinwang676/GPT-SoVITS-v3 | 2840ms | 8.03s | 0.3549 |
| 6 | **Qwen3-TTS** | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | 4846ms | 7.68s | 0.6315 |

## Detailed Results by Engine

### Supertonic

- **Model**: Supertone/supertonic-2
- **Language**: ko
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 172.9ms | 7.80s | 0.0222 |
| 2 | 33.3ms | 7.80s | 0.0043 |
| 3 | 32.5ms | 7.80s | 0.0042 |
| 4 | 32.4ms | 7.80s | 0.0042 |
| 5 | 32.4ms | 7.80s | 0.0041 |
| **Avg** | **60.7ms** | **7.80s** | **0.0078** |

### Qwen3-TTS

- **Model**: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- **Language**: ko
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 4674.3ms | 7.42s | 0.6302 |
| 2 | 5193.1ms | 8.22s | 0.6320 |
| 3 | 4472.3ms | 7.10s | 0.6302 |
| 4 | 5349.5ms | 8.46s | 0.6326 |
| 5 | 4540.0ms | 7.18s | 0.6326 |
| **Avg** | **4845.8ms** | **7.68s** | **0.6315** |

### GPT-SoVITS

- **Model**: kevinwang676/GPT-SoVITS-v3
- **Language**: ko
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 2922.1ms | 7.87s | 0.3715 |
| 2 | 2888.5ms | 8.99s | 0.3214 |
| 3 | 2800.3ms | 7.74s | 0.3619 |
| 4 | 2786.7ms | 7.50s | 0.3714 |
| 5 | 2805.0ms | 8.06s | 0.3481 |
| **Avg** | **2840.5ms** | **8.03s** | **0.3549** |

### CosyVoice

- **Model**: FunAudioLLM/CosyVoice2-0.5B
- **Language**: ko
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 7984.5ms | 23.20s | 0.3442 |
| 2 | 8133.2ms | 23.20s | 0.3506 |
| 3 | 8028.8ms | 23.20s | 0.3461 |
| 4 | 7997.6ms | 23.20s | 0.3447 |
| 5 | 8025.4ms | 23.20s | 0.3459 |
| **Avg** | **8033.9ms** | **23.20s** | **0.3463** |

### Chatterbox

- **Model**: ResembleAI/chatterbox
- **Language**: en
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 1426.5ms | 5.12s | 0.2786 |
| 2 | 1360.5ms | 4.80s | 0.2834 |
| 3 | 1469.4ms | 5.32s | 0.2762 |
| 4 | 1405.8ms | 5.00s | 0.2812 |
| 5 | 1363.9ms | 4.88s | 0.2795 |
| **Avg** | **1405.2ms** | **5.02s** | **0.2798** |

### KaniTTS

- **Model**: nineninesix/kani-tts-370m
- **Language**: ko
- **Device**: cuda

| Run | Inference Time | Audio Length | RTF |
|-----|----------------|--------------|-----|
| 1 | 2224.8ms | 7.12s | 0.3125 |
| 2 | 2313.3ms | 7.52s | 0.3076 |
| 3 | 2428.5ms | 7.92s | 0.3066 |
| 4 | 2350.6ms | 7.68s | 0.3061 |
| 5 | 2223.7ms | 7.28s | 0.3054 |
| **Avg** | **2308.2ms** | **7.50s** | **0.3076** |

## Notes

- **RTF (Real-Time Factor)**: Processing time / Audio length. RTF < 1.0 means faster than real-time.
- **TTFB**: All engines are non-streaming, so TTFB = Total Time.
- **Warm Start**: First run after warm-up may show higher latency due to JIT compilation.

## Recommendations

| Use Case | Recommended Engine | Reason |
|----------|-------------------|--------|
| Real-time dialogue | Supertonic | 60ms latency, RTF 0.008 |
| Voice cloning | CosyVoice, GPT-SoVITS | Zero-shot support |
| Korean TTS | KaniTTS, Supertonic | Native Korean support |
| English TTS | Chatterbox | High quality English |
| LLM integration | Qwen3-TTS | LLM-based, flexible |
