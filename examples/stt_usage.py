"""vTTS STT (Speech-to-Text) 사용 예제"""

import httpx

def main():
    base_url = "http://localhost:8000"
    
    # 1. 기본 전사 (Transcription)
    print("=== 1. Basic Transcription ===")
    with open("audio.wav", "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {
            "model": "large-v3",
            "language": "ko"
        }
        
        response = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        result = response.json()
        print(f"Text: {result['text']}\n")
    
    # 2. 상세 전사 (타임스탬프 포함)
    print("=== 2. Transcription with Timestamps ===")
    with open("audio.wav", "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {
            "model": "large-v3",
            "language": "ko",
            "response_format": "verbose_json",
            "timestamp_granularities": "segment,word"
        }
        
        response = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        result = response.json()
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"\nSegments:")
        for seg in result['segments'][:3]:  # 처음 3개만
            print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        print()
    
    # 3. 번역 (다른 언어 -> 영어)
    print("=== 3. Translation to English ===")
    with open("korean_audio.wav", "rb") as f:
        files = {"file": ("korean_audio.wav", f, "audio/wav")}
        data = {
            "model": "large-v3"
        }
        
        response = httpx.post(
            f"{base_url}/v1/audio/translations",
            files=files,
            data=data
        )
        
        result = response.json()
        print(f"Translated Text: {result['text']}\n")
    
    # 4. SRT 자막 파일 생성
    print("=== 4. SRT Subtitle Generation ===")
    with open("audio.wav", "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {
            "model": "large-v3",
            "language": "ko",
            "response_format": "srt"
        }
        
        response = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        srt_content = response.text
        with open("output.srt", "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        print("✓ Saved: output.srt\n")
    
    print("✓ All STT examples completed!")


if __name__ == "__main__":
    main()
