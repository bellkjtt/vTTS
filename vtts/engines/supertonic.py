"""Supertonic-2 TTS Engine - Real Implementation"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class SupertonicEngine(BaseTTSEngine):
    """Supertonic-2 TTS Engine
    
    Lightning-fast on-device TTS supporting 5 languages.
    Uses official supertonic Python package with ONNX Runtime.
    
    Supported languages: en, ko, es, pt, fr
    Voice styles: M1, M2, M3, M4, F1, F2, F3, F4
    """
    
    # 공식 voice styles (https://github.com/supertone-inc/supertonic)
    VOICE_STYLES = ["M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4"]
    
    def __init__(self, model_id: str = "Supertone/supertonic-2", **kwargs):
        super().__init__(model_id, **kwargs)
        self._supported_languages = ["en", "ko", "es", "pt", "fr"]
        self._sample_rate = 24000  # Supertonic-2 uses 24kHz
        self.tts = None
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading Supertonic model: {self.model_id}")
        logger.info("This will download ~260MB of model assets on first run")
        
        try:
            from supertonic import TTS
            
            # Supertonic TTS 초기화 (자동 다운로드)
            self.tts = TTS(auto_download=True)
            self.is_loaded = True
            
            logger.info(f"Successfully loaded Supertonic model: {self.model_id}")
            logger.info(f"Available voice styles: {', '.join(self.VOICE_STYLES)}")
            
        except ImportError as e:
            logger.error(
                "Supertonic package not installed. "
                "Install with: pip install supertonic"
            )
            raise ImportError(
                "supertonic package required. Install with: pip install supertonic"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Supertonic model: {e}")
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.tts is not None:
            del self.tts
            self.tts = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {self.model_id}")
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성을 합성합니다."""
        if not self.is_loaded:
            self.load_model()
        
        # 언어 확인
        if request.language not in self._supported_languages:
            logger.warning(
                f"Language '{request.language}' not officially supported. "
                f"Supported: {self._supported_languages}. Will attempt synthesis anyway."
            )
        
        # Voice style 매핑
        voice_name = self._map_voice_to_style(request.voice)
        logger.info(f"Using voice style: {voice_name}")
        
        try:
            # Voice style 가져오기
            voice_style = self.tts.get_voice_style(voice_name=voice_name)
            
            # 음성 합성
            logger.debug(f"Synthesizing text ({len(request.text)} chars): {request.text[:50]}...")
            wav, duration = self.tts.synthesize(
                text=request.text,
                voice_style=voice_style
            )
            
            logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
            
            # numpy 배열 형식 확인 및 변환
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            
            # float32 to float (vTTS 표준)
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # 정규화 (-1.0 ~ 1.0)
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))
            
            return TTSOutput(
                audio=wav,
                sample_rate=self._sample_rate,
                metadata={
                    "model": self.model_id,
                    "language": request.language,
                    "voice_style": voice_name,
                    "speed": request.speed,
                    "duration": duration,
                    "engine": "supertonic"
                }
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"Supertonic synthesis failed: {e}") from e
    
    def _map_voice_to_style(self, voice: str) -> str:
        """Voice ID를 Supertonic voice style로 매핑합니다."""
        # voice가 이미 valid style이면 그대로 사용
        voice_upper = voice.upper()
        if voice_upper in self.VOICE_STYLES:
            return voice_upper
        
        # 기본 매핑
        voice_lower = voice.lower()
        mapping = {
            "default": "M1",
            "male": "M1",
            "male1": "M1",
            "male2": "M2",
            "male3": "M3",
            "male4": "M4",
            "female": "F1",
            "female1": "F1",
            "female2": "F2",
            "female3": "F3",
            "female4": "F4",
        }
        
        return mapping.get(voice_lower, "M1")  # 기본값은 M1
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        # Voice styles를 소문자로도 제공
        return ["default"] + self.VOICE_STYLES + [s.lower() for s in self.VOICE_STYLES]
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False  # Supertonic-2는 배치 모드만 지원
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return False  # Supertonic-2는 사전 정의된 voice styles만 지원
