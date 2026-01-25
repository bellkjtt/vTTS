"""Bark TTS Engine for vTTS

Suno AI의 Bark 모델 - transformers 완벽 호환
다국어 지원: en, de, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh

Reference: https://huggingface.co/suno/bark
License: MIT
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger

from .base import BaseTTSEngine, TTSOutput, TTSRequest


class BarkEngine(BaseTTSEngine):
    """Bark TTS Engine
    
    Suno AI의 Bark - transformer 기반 다국어 TTS
    
    특징:
    - 13개 언어 지원 (한국어 포함)
    - 음악, 효과음 생성 가능
    - 비언어적 소리 (웃음, 기침 등) 지원
    - Voice preset 지원
    
    License: MIT
    """
    
    # 지원 언어
    SUPPORTED_LANGUAGES = [
        "en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"
    ]
    
    # Voice preset 예시 (언어별)
    VOICE_PRESETS = {
        "en": ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
               "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
               "v2/en_speaker_8", "v2/en_speaker_9"],
        "ko": ["v2/ko_speaker_0", "v2/ko_speaker_1", "v2/ko_speaker_2", "v2/ko_speaker_3",
               "v2/ko_speaker_4", "v2/ko_speaker_5", "v2/ko_speaker_6", "v2/ko_speaker_7",
               "v2/ko_speaker_8", "v2/ko_speaker_9"],
        "zh": ["v2/zh_speaker_0", "v2/zh_speaker_1", "v2/zh_speaker_2", "v2/zh_speaker_3",
               "v2/zh_speaker_4", "v2/zh_speaker_5", "v2/zh_speaker_6", "v2/zh_speaker_7",
               "v2/zh_speaker_8", "v2/zh_speaker_9"],
        "ja": ["v2/ja_speaker_0", "v2/ja_speaker_1", "v2/ja_speaker_2", "v2/ja_speaker_3",
               "v2/ja_speaker_4", "v2/ja_speaker_5", "v2/ja_speaker_6", "v2/ja_speaker_7",
               "v2/ja_speaker_8", "v2/ja_speaker_9"],
    }
    
    def __init__(
        self,
        model_id: str = "suno/bark",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = None
        self.processor = None
        self.sample_rate = 24000  # Bark default
        self._device = self._resolve_device(device)
        
    def _resolve_device(self, device: str) -> str:
        """Device 문자열 해석"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def load_model(self) -> None:
        """모델 로드"""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
            
        logger.info(f"Loading Bark model: {self.model_id}")
        logger.info("This may take a while on first run (~1.5GB download)")
        
        try:
            from transformers import AutoProcessor, BarkModel
            
            # Processor 로드
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # 모델 로드
            self.model = BarkModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Device로 이동
            self.model = self.model.to(self._device)
            
            # Sample rate 가져오기
            self.sample_rate = self.model.generation_config.sample_rate
            
            self.is_loaded = True
            logger.info(f"Successfully loaded Bark model: {self.model_id}")
            logger.info(f"Device: {self._device}")
            logger.info(f"Sample rate: {self.sample_rate} Hz")
            
        except Exception as e:
            logger.error(f"Failed to load Bark model: {e}")
            raise
            
    def unload_model(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Bark model unloaded")
        
    def _get_voice_preset(self, voice: Optional[str], language: str) -> Optional[str]:
        """Voice preset 결정"""
        if voice:
            # 직접 지정된 경우
            if voice.startswith("v2/"):
                return voice
            # 숫자만 지정된 경우 (예: "3")
            if voice.isdigit():
                lang_code = language[:2] if len(language) >= 2 else "en"
                return f"v2/{lang_code}_speaker_{voice}"
            return voice
            
        # 기본값: 언어별 speaker_0
        lang_code = language[:2] if len(language) >= 2 else "en"
        if lang_code in self.VOICE_PRESETS:
            return self.VOICE_PRESETS[lang_code][0]
        return None
        
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성 합성"""
        if not self.is_loaded:
            self.load_model()
            
        logger.info(f"Synthesizing with Bark")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # Voice preset 결정
            voice_preset = self._get_voice_preset(request.voice, request.language)
            
            if voice_preset:
                logger.debug(f"Using voice preset: {voice_preset}")
                inputs = self.processor(
                    request.text,
                    voice_preset=voice_preset,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    request.text,
                    return_tensors="pt"
                )
            
            # Device로 이동
            inputs = {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                audio_array = self.model.generate(**inputs)
            
            # numpy로 변환
            audio = audio_array.cpu().numpy().squeeze()
            
            # float32로 정규화
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # -1 ~ 1 범위로 정규화
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            logger.info(f"Synthesis complete: {len(audio)/self.sample_rate:.2f}s audio generated")
            
            return TTSOutput(
                audio=audio,
                sample_rate=self.sample_rate
            )
            
        except Exception as e:
            logger.error(f"Bark synthesis failed: {e}")
            raise RuntimeError(f"Bark synthesis failed: {e}") from e
            
    def get_available_voices(self) -> list:
        """사용 가능한 음성 목록"""
        voices = []
        for lang, presets in self.VOICE_PRESETS.items():
            voices.extend(presets)
        return voices
        
    @property
    def supported_languages(self) -> list:
        """지원 언어"""
        return self.SUPPORTED_LANGUAGES.copy()


# 엔진 등록
def register():
    from .registry import EngineRegistry
    EngineRegistry.register("bark", BarkEngine, ["suno/bark*", "*bark*"])
