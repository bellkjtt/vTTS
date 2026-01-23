# ============================================================
# 셀 3: TTS 테스트 (Python 코드)
# ============================================================
from vtts import VTTSClient
import soundfile as sf

# 위에서 출력된 ngrok URL 사용
PUBLIC_URL = "https://782807f64b8f.ngrok-free.app"  # 실제 URL로 교체

client = VTTSClient(base_url=PUBLIC_URL)

# 기본 TTS
audio = client.tts(
    text="안녕하세요, CosyVoice 음성 합성 테스트입니다.",
    voice="중문女"
)

print(f"✅ 생성 완료: {len(audio.audio)/audio.sample_rate:.2f}초")
print(f"   Sample rate: {audio.sample_rate} Hz")

# 파일 저장
audio.save("cosyvoice_test.wav")
print("✅ 저장됨: cosyvoice_test.wav")

# 재생 (로컬 환경)
try:
    import playsound
    playsound.playsound("cosyvoice_test.wav")
except ImportError:
    print("💡 재생하려면: pip install playsound")