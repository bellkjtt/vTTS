"""Qwen3-TTS Engine for vTTS"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from .base import BaseTTSEngine, TTSOutput, TTSRequest


class Qwen3TTSEngine(BaseTTSEngine):
    """Qwen3-TTS Engine
    
    Alibaba의 Qwen3-TTS 모델 지원
    - 10개 언어 지원 (중국어, 영어, 일본어, 한국어, 독일어, 프랑스어, 러시아어, 포르투갈어, 스페인어, 이탈리아어)
    - Voice Clone (Base 모델)
    - Custom Voice (9개 프리셋 음성)
    - Voice Design (음성 설명으로 생성)
    """
    
    # 모델 타입별 지원 기능
    MODEL_TYPES = {
        "CustomVoice": {"voice_clone": False, "custom_voice": True, "voice_design": False},
        "VoiceDesign": {"voice_clone": False, "custom_voice": False, "voice_design": True},
        "Base": {"voice_clone": True, "custom_voice": False, "voice_design": False},
    }
    
    # CustomVoice 모델 지원 스피커
    SUPPORTED_SPEAKERS = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee"
    ]
    
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
        "Auto"
    ]
    
    LANGUAGE_MAP = {
        "zh": "Chinese", "cn": "Chinese", "chinese": "Chinese",
        "en": "English", "english": "English",
        "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese",
        "ko": "Korean", "kr": "Korean", "korean": "Korean",
        "de": "German", "german": "German",
        "fr": "French", "french": "French",
        "ru": "Russian", "russian": "Russian",
        "pt": "Portuguese", "portuguese": "Portuguese",
        "es": "Spanish", "spanish": "Spanish",
        "it": "Italian", "italian": "Italian",
        "auto": "Auto",
    }
    
    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = None
        self.sample_rate = 24000  # Qwen3-TTS default
        self.model_type = self._detect_model_type(model_id)
        
    def _detect_model_type(self, model_id: str) -> str:
        """모델 ID에서 모델 타입 감지"""
        model_id_lower = model_id.lower()
        if "customvoice" in model_id_lower:
            return "CustomVoice"
        elif "voicedesign" in model_id_lower:
            return "VoiceDesign"
        elif "base" in model_id_lower:
            return "Base"
        else:
            # 기본값은 CustomVoice
            return "CustomVoice"
    
    def load_model(self) -> None:
        """Qwen3-TTS 모델 로드"""
        logger.info(f"Loading Qwen3-TTS model: {self.model_id}")
        logger.info(f"Model type: {self.model_type}")
        
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen-tts package not found. Install with: pip install qwen-tts"
            )
        
        # 디바이스 설정
        device_map = f"cuda:{self.device.split(':')[-1]}" if "cuda" in self.device else "cpu"
        
        # dtype 설정 (bfloat16 권장)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Loading model with device_map={device_map}, dtype={dtype}")
        
        # 모델 로드
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
        )
        
        self.is_loaded = True
        logger.info(f"Successfully loaded Qwen3-TTS model: {self.model_id}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Supported features: {self.MODEL_TYPES.get(self.model_type, {})}")
        
    def unload_model(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info(f"Unloaded model: {self.model_id}")
    
    def _map_language(self, lang: str) -> str:
        """언어 코드를 Qwen3-TTS 형식으로 변환"""
        lang_lower = lang.lower()
        return self.LANGUAGE_MAP.get(lang_lower, "Auto")
    
    def _map_voice_to_speaker(self, voice: str) -> str:
        """voice를 speaker로 매핑"""
        # 직접 스피커 이름인 경우
        if voice in self.SUPPORTED_SPEAKERS:
            return voice
        
        # 소문자로 매칭 시도
        voice_lower = voice.lower()
        for speaker in self.SUPPORTED_SPEAKERS:
            if speaker.lower() == voice_lower:
                return speaker
        
        # 기본값
        return "Vivian"
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성 합성"""
        if not self.is_loaded:
            self.load_model()
        
        text = request.text
        language = self._map_language(request.language)
        extra = request.extra_params or {}
        
        logger.info(f"Synthesizing with Qwen3-TTS ({self.model_type})")
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"Language: {language}")
        
        try:
            if self.model_type == "CustomVoice":
                audio = self._synthesize_custom_voice(text, language, request, extra)
            elif self.model_type == "VoiceDesign":
                audio = self._synthesize_voice_design(text, language, request, extra)
            elif self.model_type == "Base":
                audio = self._synthesize_voice_clone(text, language, request, extra)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            return TTSOutput(
                audio=audio,
                sample_rate=self.sample_rate,
                metadata={
                    "model_type": self.model_type,
                    "language": language,
                }
            )
            
        except Exception as e:
            logger.error(f"Qwen3-TTS synthesis failed: {e}")
            raise RuntimeError(f"Qwen3-TTS synthesis failed: {e}") from e
    
    def _synthesize_custom_voice(
        self,
        text: str,
        language: str,
        request: TTSRequest,
        extra: Dict[str, Any]
    ) -> np.ndarray:
        """CustomVoice 모델로 합성"""
        speaker = self._map_voice_to_speaker(request.voice)
        instruct = extra.get("instruct", "")
        
        logger.info(f"Using CustomVoice with speaker={speaker}")
        
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct else None,
        )
        
        self.sample_rate = sr
        return wavs[0]
    
    def _synthesize_voice_design(
        self,
        text: str,
        language: str,
        request: TTSRequest,
        extra: Dict[str, Any]
    ) -> np.ndarray:
        """VoiceDesign 모델로 합성"""
        instruct = extra.get("instruct", "")
        if not instruct:
            instruct = "Natural and clear voice."
        
        logger.info(f"Using VoiceDesign with instruct={instruct[:50]}...")
        
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        
        self.sample_rate = sr
        return wavs[0]
    
    def _synthesize_voice_clone(
        self,
        text: str,
        language: str,
        request: TTSRequest,
        extra: Dict[str, Any]
    ) -> np.ndarray:
        """Base 모델로 Voice Clone"""
        ref_audio = request.reference_audio
        ref_text = request.reference_text
        
        if ref_audio is None:
            raise ValueError("Base model requires reference_audio for voice cloning")
        
        logger.info(f"Using Voice Clone with ref_audio={ref_audio}")
        
        # reference_audio 처리
        if isinstance(ref_audio, np.ndarray):
            # numpy array인 경우 튜플로 변환
            ref_audio = (ref_audio, self.sample_rate)
        
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        
        self.sample_rate = sr
        return wavs[0]
    
    @property
    def supported_languages(self) -> List[str]:
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supported_voices(self) -> List[str]:
        if self.model_type == "CustomVoice":
            return self.SUPPORTED_SPEAKERS
        else:
            return ["reference"]
    
    @property
    def default_sample_rate(self) -> int:
        return self.sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        return True  # Qwen3-TTS supports streaming
    
    @property
    def supports_zero_shot(self) -> bool:
        return self.model_type == "Base"
