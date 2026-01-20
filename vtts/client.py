"""vTTS Client"""

from pathlib import Path
from typing import Optional, Union

import httpx
import numpy as np
from loguru import logger

from vtts.utils.audio import decode_audio


class AudioResponse:
    """오디오 응답"""
    
    def __init__(self, audio_bytes: bytes, format: str = "mp3"):
        self.audio_bytes = audio_bytes
        self.format = format
        self._audio_data = None
        self._sample_rate = None
    
    @property
    def audio(self) -> np.ndarray:
        """오디오 데이터 (numpy array)"""
        if self._audio_data is None:
            self._audio_data, self._sample_rate = decode_audio(self.audio_bytes)
        return self._audio_data
    
    @property
    def sample_rate(self) -> int:
        """샘플링 레이트"""
        if self._sample_rate is None:
            self._audio_data, self._sample_rate = decode_audio(self.audio_bytes)
        return self._sample_rate
    
    def save(self, path: Union[str, Path]) -> None:
        """파일로 저장"""
        path = Path(path)
        path.write_bytes(self.audio_bytes)
        logger.info(f"Saved audio to: {path}")


class VTTSClient:
    """vTTS 클라이언트"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0
    ):
        """
        Args:
            base_url: vTTS 서버 URL
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "default",
        language: str = "ko",
        speed: float = 1.0,
        response_format: str = "mp3",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None
    ) -> AudioResponse:
        """
        텍스트를 음성으로 변환합니다.
        
        Args:
            text: 합성할 텍스트
            model: 모델 ID (None이면 서버의 기본 모델)
            voice: 음성 ID
            language: 언어 코드 (ko, en, ja, etc)
            speed: 속도 (0.25 ~ 4.0)
            response_format: 응답 포맷 (mp3, wav, flac, etc)
            reference_audio: 참조 오디오 (zero-shot용)
            reference_text: 참조 텍스트
            
        Returns:
            AudioResponse: 오디오 응답
        """
        # 모델 자동 감지
        if model is None:
            model = self._get_default_model()
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "language": language,
            "speed": speed,
            "response_format": response_format,
        }
        
        if reference_audio:
            payload["reference_audio"] = reference_audio
        if reference_text:
            payload["reference_text"] = reference_text
        
        logger.info(f"Synthesizing: {text[:50]}...")
        
        response = self.client.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload
        )
        response.raise_for_status()
        
        return AudioResponse(response.content, format=response_format)
    
    def _get_default_model(self) -> str:
        """기본 모델 가져오기"""
        response = self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
        if not models:
            raise ValueError("No models available")
        return models[0]["id"]
    
    def list_models(self) -> list[dict]:
        """사용 가능한 모델 목록"""
        response = self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"]
    
    def list_voices(self) -> list[dict]:
        """사용 가능한 음성 목록"""
        response = self.client.get(f"{self.base_url}/v1/voices")
        response.raise_for_status()
        return response.json()["voices"]
    
    def health(self) -> dict:
        """헬스 체크"""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def close(self) -> None:
        """클라이언트 종료"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
