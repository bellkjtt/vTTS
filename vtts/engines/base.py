"""Base TTS Engine Interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch


@dataclass
class TTSOutput:
    """TTS 출력 결과"""
    audio: np.ndarray  # 오디오 데이터 (샘플)
    sample_rate: int  # 샘플링 레이트
    metadata: Optional[Dict[str, Any]] = None  # 추가 메타데이터


@dataclass
class TTSRequest:
    """TTS 요청"""
    text: str  # 입력 텍스트
    language: str = "ko"  # 언어 코드
    voice: str = "default"  # 음성 ID
    speed: float = 1.0  # 속도 (0.5 ~ 2.0)
    reference_audio: Optional[Union[str, Path, np.ndarray]] = None  # 참조 오디오 (zero-shot)
    reference_text: Optional[str] = None  # 참조 텍스트
    stream: bool = False  # 스트리밍 여부
    
    # 추가 엔진별 파라미터
    extra_params: Optional[Dict[str, Any]] = None


class BaseTTSEngine(ABC):
    """TTS 엔진 베이스 클래스"""
    
    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model_id: Huggingface 모델 ID
            device: 디바이스 (cuda, cpu, etc)
            cache_dir: 모델 캐시 디렉토리
            **kwargs: 추가 엔진별 파라미터
        """
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """모델을 로드합니다."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        pass
    
    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """
        음성을 합성합니다.
        
        Args:
            request: TTS 요청
            
        Returns:
            TTSOutput: 합성된 오디오
        """
        pass
    
    def synthesize_stream(self, request: TTSRequest) -> Generator[TTSOutput, None, None]:
        """
        스트리밍으로 음성을 합성합니다.
        
        Args:
            request: TTS 요청
            
        Yields:
            TTSOutput: 합성된 오디오 청크
        """
        # 기본 구현: 전체 오디오를 한번에 반환
        yield self.synthesize(request)
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """지원하는 언어 코드 목록"""
        pass
    
    @property
    @abstractmethod
    def supported_voices(self) -> List[str]:
        """지원하는 음성 ID 목록"""
        pass
    
    @property
    @abstractmethod
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        pass
    
    @property
    @abstractmethod
    def supports_zero_shot(self) -> bool:
        """Zero-shot 음성 복제 지원 여부"""
        pass
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        # 일반적으로 unload하지 않음 (캐싱 유지)
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_languages": self.supported_languages,
            "supported_voices": self.supported_voices,
            "sample_rate": self.default_sample_rate,
            "supports_streaming": self.supports_streaming,
            "supports_zero_shot": self.supports_zero_shot,
        }
