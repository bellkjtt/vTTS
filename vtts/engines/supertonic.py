"""Supertonic TTS Engine"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class SupertonicEngine(BaseTTSEngine):
    """Supertonic-2 TTS Engine
    
    Lightning-fast on-device TTS supporting 5 languages.
    """
    
    def __init__(self, model_id: str = "Supertone/supertonic-2", **kwargs):
        super().__init__(model_id, **kwargs)
        self._supported_languages = ["en", "ko", "es", "pt", "fr"]
        self._sample_rate = 24000
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading Supertonic model: {self.model_id}")
        
        try:
            # Supertonic 모델 로드
            # Note: 실제 구현은 supertonic 패키지에 따라 달라집니다
            from supertonic import Supertonic
            
            # 모델 파일 다운로드
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model.onnx",  # 실제 파일명 확인 필요
                cache_dir=self.cache_dir
            )
            
            self.model = Supertonic.from_pretrained(
                model_path,
                device=self.device
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded Supertonic model: {self.model_id}")
            
        except ImportError:
            logger.error("Supertonic package not installed. Install with: pip install supertonic")
            raise
        except Exception as e:
            logger.error(f"Failed to load Supertonic model: {e}")
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {self.model_id}")
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성을 합성합니다."""
        if not self.is_loaded:
            self.load_model()
        
        # 언어 확인
        if request.language not in self._supported_languages:
            logger.warning(
                f"Language '{request.language}' not in supported list {self._supported_languages}, "
                "but will attempt synthesis"
            )
        
        # Supertonic 추론
        # Note: 실제 API는 supertonic 패키지 문서 확인 필요
        audio = self.model.synthesize(
            text=request.text,
            language=request.language,
            speed=request.speed
        )
        
        # numpy 배열로 변환
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        return TTSOutput(
            audio=audio,
            sample_rate=self._sample_rate,
            metadata={
                "model": self.model_id,
                "language": request.language,
                "speed": request.speed
            }
        )
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        return ["default"]  # Supertonic은 단일 음성
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False  # Supertonic은 스트리밍 미지원
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return False  # Supertonic은 zero-shot 미지원
