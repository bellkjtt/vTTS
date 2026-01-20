"""OpenAI SDK 호환성 예제"""

from openai import OpenAI

def main():
    # OpenAI 클라이언트로 vTTS 사용
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"  # vTTS는 API 키 불필요
    )
    
    # TTS 요청 (OpenAI와 동일한 API)
    response = client.audio.speech.create(
        model="auto",  # 서버의 기본 모델 사용
        voice="default",
        input="안녕하세요, OpenAI SDK로 vTTS를 사용하고 있습니다."
    )
    
    # 파일로 저장
    response.stream_to_file("openai_compatible.mp3")
    print("✓ Saved: openai_compatible.mp3")
    
    # 영어 예제
    response = client.audio.speech.create(
        model="auto",
        voice="default",
        input="This is an example of using vTTS with OpenAI SDK."
    )
    response.stream_to_file("openai_english.mp3")
    print("✓ Saved: openai_english.mp3")


if __name__ == "__main__":
    main()
