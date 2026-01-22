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
        reference_text: Optional[str] = None,
        # Supertonic 엔진 파라미터
        total_steps: Optional[int] = None,
        silence_duration: Optional[float] = None,
        # GPT-SoVITS 엔진 파라미터
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        sample_steps: Optional[int] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        text_split_method: Optional[str] = None,
        batch_size: Optional[int] = None,
        fragment_interval: Optional[float] = None,
        parallel_infer: Optional[bool] = None,
        # 범용 파라미터
        extra_params: Optional[dict] = None
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
            reference_audio: 참조 오디오 (zero-shot용, GPT-SoVITS 필수)
            reference_text: 참조 텍스트 (GPT-SoVITS 필수)
            
            # Supertonic 전용
            total_steps: Denoising steps (기본값: 5, 높을수록 품질↑ 속도↓)
            silence_duration: 청크 사이 무음 시간 (초, 기본값: 0.3)
            
            # GPT-SoVITS 전용
            top_k: Top-K 샘플링 (기본값: 15)
            top_p: Top-P 샘플링 (기본값: 1.0)
            temperature: 생성 다양성 (기본값: 1.0, 높을수록 다양)
            sample_steps: 샘플링 스텝 수 (기본값: 32 for v3)
            seed: 랜덤 시드 (기본값: -1, 재현성 위해 설정)
            repetition_penalty: 반복 억제 (기본값: 1.35, 높을수록 반복 감소)
            text_split_method: 텍스트 분할 방식 (기본값: cut5)
            batch_size: 배치 크기 (기본값: 1)
            fragment_interval: 문장 조각 간 간격 초 (기본값: 0.3)
            parallel_infer: 병렬 추론 활성화 (기본값: True)
            
            # 범용
            extra_params: 엔진별 추가 파라미터 딕셔너리
            
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
        
        # Supertonic 파라미터
        if total_steps is not None:
            payload["total_steps"] = total_steps
        if silence_duration is not None:
            payload["silence_duration"] = silence_duration
        
        # GPT-SoVITS 파라미터
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if temperature is not None:
            payload["temperature"] = temperature
        if sample_steps is not None:
            payload["sample_steps"] = sample_steps
        if seed is not None:
            payload["seed"] = seed
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if text_split_method is not None:
            payload["text_split_method"] = text_split_method
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if fragment_interval is not None:
            payload["fragment_interval"] = fragment_interval
        if parallel_infer is not None:
            payload["parallel_infer"] = parallel_infer
        
        # 추가 파라미터 (extra_params로 전달된 것들)
        if extra_params:
            payload.update(extra_params)
        
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
