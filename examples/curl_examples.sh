#!/bin/bash
# vTTS cURL 사용 예제

BASE_URL="http://localhost:8000"

echo "=== vTTS cURL Examples ==="

# 1. 헬스 체크
echo -e "\n1. Health Check"
curl -s "$BASE_URL/health" | jq

# 2. 모델 목록
echo -e "\n2. List Models"
curl -s "$BASE_URL/v1/models" | jq

# 3. 음성 목록
echo -e "\n3. List Voices"
curl -s "$BASE_URL/v1/voices" | jq

# 4. 기본 TTS (한국어)
echo -e "\n4. Korean TTS"
curl -s "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "안녕하세요, vTTS입니다.",
    "voice": "default",
    "language": "ko"
  }' \
  --output korean.mp3
echo "✓ Saved: korean.mp3"

# 5. 영어 TTS
echo -e "\n5. English TTS"
curl -s "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "Hello, this is vTTS.",
    "voice": "default",
    "language": "en"
  }' \
  --output english.mp3
echo "✓ Saved: english.mp3"

# 6. 속도 조절
echo -e "\n6. Speed Control"
curl -s "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "빠른 속도로 말하기",
    "voice": "default",
    "language": "ko",
    "speed": 1.5
  }' \
  --output fast.mp3
echo "✓ Saved: fast.mp3"

# 7. WAV 포맷
echo -e "\n7. WAV Format"
curl -s "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "WAV 포맷으로 저장",
    "voice": "default",
    "language": "ko",
    "response_format": "wav"
  }' \
  --output output.wav
echo "✓ Saved: output.wav"

echo -e "\n=== All examples completed! ==="
