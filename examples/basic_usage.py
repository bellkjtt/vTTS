"""vTTS 기본 사용 예제"""

from vtts import VTTSClient

def main():
    # 클라이언트 생성
    client = VTTSClient(base_url="http://localhost:8000")
    
    # 헬스 체크
    print("Health:", client.health())
    
    # 사용 가능한 모델 확인
    print("\nModels:", client.list_models())
    
    # 기본 TTS
    audio = client.tts(
        text="안녕하세요, vTTS를 사용해주셔서 감사합니다.",
        language="ko"
    )
    audio.save("output_korean.mp3")
    print("✓ Saved: output_korean.mp3")
    
    # 영어 TTS
    audio = client.tts(
        text="Hello, thank you for using vTTS.",
        language="en"
    )
    audio.save("output_english.mp3")
    print("✓ Saved: output_english.mp3")
    
    # 속도 조절
    audio = client.tts(
        text="빠른 속도로 말하기",
        language="ko",
        speed=1.5
    )
    audio.save("output_fast.mp3")
    print("✓ Saved: output_fast.mp3")
    
    # WAV 포맷
    audio = client.tts(
        text="WAV 포맷으로 저장",
        language="ko",
        response_format="wav"
    )
    audio.save("output.wav")
    print("✓ Saved: output.wav")
    
    print("\n✓ All examples completed!")


if __name__ == "__main__":
    main()
