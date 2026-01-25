"""Parler-TTS Engine for vTTS

Parler-TTS - 텍스트 설명 기반 음성 제어 TTS
음성 특성을 자연어로 설명하여 제어 가능

Reference: https://huggingface.co/parler-tts/parler-tts-mini-v1
License: Apache 2.0
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger

from .base import BaseTTSEngine, TTSOutput, TTSRequest


class ParlerTTSEngine(BaseTTSEngine):
    """Parler-TTS Engine
    
    텍스트 설명 기반 음성 제어 TTS
    
    특징:
    - 음성 특성을 자연어로 설명하여 제어
    - 고품질 영어 음성
    - transformers 완벽 호환
    
    License: Apache 2.0
    """
    
    # 기본 음성 설명 프리셋
    VOICE_DESCRIPTIONS = {
        "default": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch.",
        "male": "A male speaker with a calm and steady voice speaks at a moderate pace.",
        "female": "A female speaker with a clear and pleasant voice speaks naturally.",
        "narrator": "A professional narrator speaks clearly with good articulation and moderate pace.",
        "expressive": "An expressive speaker delivers the speech with enthusiasm and varied intonation.",
        "calm": "A calm and soothing voice speaks slowly and clearly.",
        "fast": "A speaker delivers the speech quickly but clearly.",
        "slow": "A speaker delivers the speech slowly with clear pronunciation.",
    }
    
    SUPPORTED_LANGUAGES = ["en"]
    
    def __init__(
        self,
        model_id: str = "parler-tts/parler-tts-mini-v1",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = None
        self.tokenizer = None
        self.sample_rate = 44100  # Parler-TTS default
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
            
        logger.info(f"Loading Parler-TTS model: {self.model_id}")
        logger.info("This may take a while on first run (~2GB download)")
        
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            
            # Tokenizer 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # 모델 로드
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Device로 이동
            self.model = self.model.to(self._device)
            
            # Sample rate 가져오기
            self.sample_rate = self.model.config.sampling_rate
            
            self.is_loaded = True
            logger.info(f"Successfully loaded Parler-TTS model: {self.model_id}")
            logger.info(f"Device: {self._device}")
            logger.info(f"Sample rate: {self.sample_rate} Hz")
            
        except ImportError:
            # parler_tts 패키지 미설치 시 설치
            logger.warning("parler-tts package not found. Installing...")
            import subprocess
            import sys
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "parler-tts", "-q"],
                check=True
            )
            # 재시도
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to load Parler-TTS model: {e}")
            raise
            
    def unload_model(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Parler-TTS model unloaded")
        
    def _get_voice_description(self, voice: Optional[str]) -> str:
        """Voice description 결정"""
        if voice is None:
            return self.VOICE_DESCRIPTIONS["default"]
            
        # 프리셋 이름인 경우
        if voice.lower() in self.VOICE_DESCRIPTIONS:
            return self.VOICE_DESCRIPTIONS[voice.lower()]
            
        # 직접 설명을 전달한 경우 (긴 문자열)
        if len(voice) > 20:
            return voice
            
        # 알 수 없는 경우 기본값
        return self.VOICE_DESCRIPTIONS["default"]
        
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성 합성"""
        if not self.is_loaded:
            self.load_model()
            
        logger.info(f"Synthesizing with Parler-TTS")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # Voice description 결정
            description = self._get_voice_description(request.voice)
            logger.debug(f"Voice description: {description[:50]}...")
            
            # 입력 토큰화
            input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self._device)
            prompt_input_ids = self.tokenizer(request.text, return_tensors="pt").input_ids.to(self._device)
            
            # 생성
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids
                )
            
            # numpy로 변환
            audio = generation.cpu().numpy().squeeze()
            
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
            logger.error(f"Parler-TTS synthesis failed: {e}")
            raise RuntimeError(f"Parler-TTS synthesis failed: {e}") from e
            
    def get_available_voices(self) -> list:
        """사용 가능한 음성 프리셋 목록"""
        return list(self.VOICE_DESCRIPTIONS.keys())
        
    @property
    def supported_languages(self) -> list:
        """지원 언어"""
        return self.SUPPORTED_LANGUAGES.copy()
        
    @property
    def supported_voices(self) -> list:
        """지원 음성 목록"""
        return self.get_available_voices()
        
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self.sample_rate
        
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False
        
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 음성 복제 지원 여부"""
        return False  # 텍스트 설명 기반 제어


# 엔진 등록
def register():
    from .registry import EngineRegistry
    EngineRegistry.register("parler", ParlerTTSEngine, ["parler-tts/*", "*parler*"])
