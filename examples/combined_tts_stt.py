"""vTTS TTS + STT 통합 사용 예제"""

import httpx
from pathlib import Path

def main():
    """TTS로 음성 생성 후 STT로 다시 인식하는 Round-trip 테스트"""
    
    base_url = "http://localhost:8000"
    client = httpx.Client(base_url=base_url, timeout=60.0)
    
    # 1. TTS: 텍스트 -> 음성
    print("=== Step 1: Text-to-Speech ===")
    original_text = "안녕하세요, 이것은 vTTS 테스트입니다. 음성 합성과 인식이 모두 잘 작동하는지 확인하고 있습니다."
    print(f"Original: {original_text}\n")
    
    tts_response = client.post(
        "/v1/audio/speech",
        json={
            "model": "auto",
            "input": original_text,
            "language": "ko",
            "response_format": "wav"
        }
    )
    
    # 음성 파일 저장
    audio_path = Path("roundtrip_test.wav")
    audio_path.write_bytes(tts_response.content)
    print(f"✓ Generated audio: {audio_path}")
    print(f"  Size: {len(tts_response.content)} bytes\n")
    
    # 2. STT: 음성 -> 텍스트
    print("=== Step 2: Speech-to-Text ===")
    
    with open(audio_path, "rb") as f:
        files = {"file": (str(audio_path), f, "audio/wav")}
        data = {
            "model": "large-v3",
            "language": "ko",
            "response_format": "verbose_json"
        }
        
        stt_response = client.post(
            "/v1/audio/transcriptions",
            files=files,
            data=data
        )
    
    result = stt_response.json()
    transcribed_text = result["text"]
    
    print(f"Transcribed: {transcribed_text}")
    print(f"Language: {result['language']}")
    print(f"Duration: {result['duration']:.2f}s\n")
    
    # 3. 결과 비교
    print("=== Step 3: Comparison ===")
    print(f"Original:    {original_text}")
    print(f"Transcribed: {transcribed_text}")
    
    # 간단한 유사도 체크
    original_words = set(original_text.replace(",", "").replace(".", "").split())
    transcribed_words = set(transcribed_text.replace(",", "").replace(".", "").split())
    
    overlap = len(original_words & transcribed_words)
    similarity = overlap / max(len(original_words), len(transcribed_words)) * 100
    
    print(f"\nWord overlap: {overlap}/{len(original_words)}")
    print(f"Similarity: {similarity:.1f}%")
    
    if similarity > 80:
        print("\n✅ Round-trip test PASSED!")
    else:
        print("\n⚠️ Round-trip test shows some differences")
    
    # 정리
    audio_path.unlink()
    print(f"\n🧹 Cleaned up: {audio_path}")


if __name__ == "__main__":
    main()
