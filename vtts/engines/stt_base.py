"""Base STT Engine Interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch


@dataclass
class STTOutput:
    """STT 출력 결과"""
    text: str  # 인식된 텍스트
    language: Optional[str] = None  # 감지된 언어
    segments: Optional[List[Dict[str, Any]]] = None  # 세그먼트 정보 (타임스탬프 등)
    metadata: Optional[Dict[str, Any]] = None  # 추가 메타데이터


@dataclass
class STTRequest:
    """STT 요청"""
    audio: Union[str, Path, np.ndarray, bytes]  # 오디오 데이터
    language: Optional[str] = None  # 언어 힌트 (None이면 자동 감지)
    task: str = "transcribe"  # transcribe 또는 translate
    temperature: float = 0.0  # 샘플링 온도
    timestamp_granularities: List[str] = None  # ["word", "segment"]
    response_format: str = "json"  # json, text, srt, vtt, verbose_json
    
    # 추가 엔진별 파라미터
    extra_params: Optional[Dict[str, Any]] = None


class BaseSTTEngine(ABC):
    """STT 엔진 베이스 클래스"""
    
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
    def transcribe(self, request: STTRequest) -> STTOutput:
        """
        음성을 텍스트로 변환합니다.
        
        Args:
            request: STT 요청
            
        Returns:
            STTOutput: 인식된 텍스트
        """
        pass
    
    def transcribe_stream(self, request: STTRequest) -> Generator[STTOutput, None, None]:
        """
        스트리밍으로 음성을 인식합니다.
        
        Args:
            request: STT 요청
            
        Yields:
            STTOutput: 인식된 텍스트 청크
        """
        # 기본 구현: 전체 텍스트를 한번에 반환
        yield self.transcribe(request)
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """지원하는 언어 코드 목록"""
        pass
    
    @property
    @abstractmethod
    def supports_translation(self) -> bool:
        """번역 지원 여부 (음성을 영어로)"""
        pass
    
    @property
    @abstractmethod
    def supports_timestamps(self) -> bool:
        """타임스탬프 지원 여부"""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        pass
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_languages": self.supported_languages,
            "supports_translation": self.supports_translation,
            "supports_timestamps": self.supports_timestamps,
            "supports_streaming": self.supports_streaming,
        }
