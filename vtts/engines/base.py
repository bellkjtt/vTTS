"""Base TTS Engine Interface"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from loguru import logger


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


class ReferenceAudioCache:
    """Reference audio 캐싱 클래스"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        # 오디오 데이터 캐시: key -> (audio_data, sample_rate)
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        # 엔진별 feature 캐시: key -> engine_features
        self._feature_cache: Dict[str, Any] = {}
        # LRU 순서 추적
        self._access_order: List[str] = []
    
    def _get_cache_key(self, ref_audio: Union[str, Path, np.ndarray], ref_text: Optional[str] = None) -> str:
        """캐시 키 생성"""
        if isinstance(ref_audio, (str, Path)):
            key_str = str(ref_audio)
        elif isinstance(ref_audio, np.ndarray):
            # numpy array의 해시 생성
            key_str = hashlib.md5(ref_audio.tobytes()).hexdigest()
        else:
            key_str = str(id(ref_audio))
        
        if ref_text:
            key_str = f"{key_str}:{ref_text}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_access_order(self, key: str):
        """LRU 순서 업데이트"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # 최대 크기 초과 시 가장 오래된 항목 제거
        while len(self._access_order) > self.max_size:
            old_key = self._access_order.pop(0)
            self._audio_cache.pop(old_key, None)
            self._feature_cache.pop(old_key, None)
    
    def get_audio(self, ref_audio: Union[str, Path, np.ndarray], ref_text: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        """캐시에서 오디오 데이터 반환"""
        key = self._get_cache_key(ref_audio, ref_text)
        if key in self._audio_cache:
            self._update_access_order(key)
            logger.debug(f"Reference audio cache HIT: {key[:8]}...")
            return self._audio_cache[key]
        return None
    
    def set_audio(self, ref_audio: Union[str, Path, np.ndarray], audio_data: np.ndarray, sample_rate: int, ref_text: Optional[str] = None):
        """오디오 데이터 캐시에 저장"""
        key = self._get_cache_key(ref_audio, ref_text)
        self._audio_cache[key] = (audio_data, sample_rate)
        self._update_access_order(key)
        logger.debug(f"Reference audio cached: {key[:8]}...")
    
    def get_features(self, ref_audio: Union[str, Path, np.ndarray], ref_text: Optional[str] = None) -> Optional[Any]:
        """캐시에서 엔진별 feature 반환"""
        key = self._get_cache_key(ref_audio, ref_text)
        if key in self._feature_cache:
            self._update_access_order(key)
            logger.debug(f"Reference feature cache HIT: {key[:8]}...")
            return self._feature_cache[key]
        return None
    
    def set_features(self, ref_audio: Union[str, Path, np.ndarray], features: Any, ref_text: Optional[str] = None):
        """엔진별 feature 캐시에 저장"""
        key = self._get_cache_key(ref_audio, ref_text)
        self._feature_cache[key] = features
        self._update_access_order(key)
        logger.debug(f"Reference features cached: {key[:8]}...")
    
    def load_audio(self, ref_audio: Union[str, Path, np.ndarray], ref_text: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Reference audio 로드 (캐시 우선)
        
        Returns:
            (audio_data, sample_rate)
        """
        # 캐시 확인
        cached = self.get_audio(ref_audio, ref_text)
        if cached is not None:
            return cached
        
        # 파일/URL에서 로드
        if isinstance(ref_audio, (str, Path)):
            ref_path = str(ref_audio)
            logger.info(f"Loading reference audio: {ref_path}")
            
            if ref_path.startswith(('http://', 'https://')):
                # URL에서 다운로드
                import tempfile
                import urllib.request
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    urllib.request.urlretrieve(ref_path, tmp.name)
                    audio_data, sample_rate = sf.read(tmp.name)
            else:
                # 로컬 파일
                audio_data, sample_rate = sf.read(ref_path)
            
            # 캐시에 저장
            self.set_audio(ref_audio, audio_data, sample_rate, ref_text)
            return audio_data, sample_rate
        
        elif isinstance(ref_audio, np.ndarray):
            # 이미 numpy array인 경우
            return ref_audio, 24000  # 기본 sample rate
        
        elif isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            # (audio_data, sample_rate) 튜플
            return ref_audio
        
        raise ValueError(f"Unsupported reference_audio type: {type(ref_audio)}")
    
    def clear(self):
        """캐시 초기화"""
        self._audio_cache.clear()
        self._feature_cache.clear()
        self._access_order.clear()
        logger.info("Reference audio cache cleared")


class BaseTTSEngine(ABC):
    """TTS 엔진 베이스 클래스"""
    
    # 모든 엔진이 공유하는 reference audio 캐시
    _ref_cache: Optional[ReferenceAudioCache] = None
    
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
        
        # Reference audio 캐시 초기화 (클래스 레벨 싱글톤)
        if BaseTTSEngine._ref_cache is None:
            BaseTTSEngine._ref_cache = ReferenceAudioCache(max_size=20)
    
    @property
    def ref_cache(self) -> ReferenceAudioCache:
        """Reference audio 캐시"""
        return BaseTTSEngine._ref_cache
        
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
