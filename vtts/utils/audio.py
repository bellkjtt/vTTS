"""Audio Processing Utilities"""

import io
from typing import Literal

import numpy as np
import soundfile as sf
from pydub import AudioSegment


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
) -> bytes:
    """
    오디오를 원하는 포맷으로 인코딩합니다.
    
    Args:
        audio: 오디오 데이터 (numpy array)
        sample_rate: 샘플링 레이트
        format: 출력 포맷
        
    Returns:
        인코딩된 오디오 바이트
    """
    # numpy array를 16-bit PCM으로 변환
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    
    if format == "pcm":
        return audio.tobytes()
    
    # WAV로 먼저 변환
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sample_rate, format="WAV")
    wav_buffer.seek(0)
    
    if format == "wav":
        return wav_buffer.read()
    
    # pydub으로 다른 포맷 변환
    audio_segment = AudioSegment.from_wav(wav_buffer)
    
    output_buffer = io.BytesIO()
    audio_segment.export(output_buffer, format=format)
    output_buffer.seek(0)
    
    return output_buffer.read()


def decode_audio(
    audio_bytes: bytes,
    format: str = "auto"
) -> tuple[np.ndarray, int]:
    """
    오디오 바이트를 디코딩합니다.
    
    Args:
        audio_bytes: 오디오 바이트
        format: 오디오 포맷 (auto이면 자동 감지)
        
    Returns:
        (audio_data, sample_rate)
    """
    buffer = io.BytesIO(audio_bytes)
    
    if format == "auto":
        # soundfile로 자동 감지
        audio, sr = sf.read(buffer)
    else:
        audio, sr = sf.read(buffer, format=format)
    
    return audio, sr
