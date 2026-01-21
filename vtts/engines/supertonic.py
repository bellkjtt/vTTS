"""Supertonic-2 TTS Engine - ONNX Direct Inference

다국어 지원 (en, ko, es, pt, fr)을 위해 내장된 ONNX 추론 모듈 사용.
PyPI supertonic 패키지 대신 GitHub 소스 기반 구현.

Reference: https://github.com/supertone-inc/supertonic
Model: https://huggingface.co/Supertone/supertonic-2
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class SupertonicEngine(BaseTTSEngine):
    """Supertonic-2 TTS Engine
    
    Lightning-fast on-device TTS via ONNX Runtime.
    
    Supported languages: en (English), ko (Korean), es (Spanish), pt (Portuguese), fr (French)
    Voice styles: M1, M2, M3, M4, F1, F2, F3, F4
    
    Uses built-in ONNX inference module for full multilingual support.
    
    Reference: https://github.com/supertone-inc/supertonic
    """
    
    # 공식 voice styles
    VOICE_STYLES = ["M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4"]
    
    # Supertonic 2는 5개 언어 지원
    SUPPORTED_LANGS = ["en", "ko", "es", "pt", "fr"]
    
    # HuggingFace 모델 ID
    HF_MODEL_ID = "Supertone/supertonic-2"
    
    def __init__(self, model_id: str = "Supertone/supertonic-2", **kwargs):
        super().__init__(model_id, **kwargs)
        self._supported_languages = self.SUPPORTED_LANGS
        self._sample_rate = None  # 모델 로드 후 설정
        self.tts = None
        self._model_dir = None
        self._voice_styles_cache = {}
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading Supertonic model: {self.model_id}")
        logger.info("This will download ~260MB of model assets on first run")
        
        try:
            # HuggingFace에서 모델 다운로드
            logger.info("Downloading model from HuggingFace...")
            self._model_dir = snapshot_download(
                repo_id=self.HF_MODEL_ID,
                allow_patterns=["onnx/*", "voice_styles/*", "config.json"],
            )
            logger.info(f"Model downloaded to: {self._model_dir}")
            
            # ONNX 디렉토리 경로
            onnx_dir = os.path.join(self._model_dir, "onnx")
            
            # 내장 모듈로 TTS 로드
            from vtts.engines._supertonic.helper import load_text_to_speech
            
            self.tts = load_text_to_speech(onnx_dir, use_gpu=False)
            self.is_loaded = True
            
            # 실제 샘플레이트 가져오기 (모델 설정에서)
            self._sample_rate = self.tts.sample_rate
            
            logger.info(f"Successfully loaded Supertonic model: {self.model_id}")
            logger.info(f"Sample rate: {self._sample_rate} Hz")
            logger.info(f"Available voice styles: {', '.join(self.VOICE_STYLES)}")
            logger.info(f"Supported languages: {', '.join(self.SUPPORTED_LANGS)}")
            
        except ImportError as e:
            logger.error(f"Failed to import ONNX modules: {e}")
            logger.error("Install with: pip install onnxruntime")
            raise ImportError(
                "onnxruntime package required. Install with: pip install onnxruntime"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Supertonic model: {e}")
            raise
    
    def _get_voice_style(self, voice_name: str):
        """Voice style을 로드합니다."""
        if voice_name in self._voice_styles_cache:
            return self._voice_styles_cache[voice_name]
        
        from vtts.engines._supertonic.helper import load_voice_style
        
        voice_style_path = os.path.join(
            self._model_dir, "voice_styles", f"{voice_name}.json"
        )
        
        if not os.path.exists(voice_style_path):
            logger.warning(f"Voice style {voice_name} not found, using M1")
            voice_style_path = os.path.join(
                self._model_dir, "voice_styles", "M1.json"
            )
        
        style = load_voice_style([voice_style_path], verbose=False)
        self._voice_styles_cache[voice_name] = style
        return style
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.tts is not None:
            del self.tts
            self.tts = None
            self._voice_styles_cache.clear()
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {self.model_id}")
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성을 합성합니다.
        
        Supertonic 2는 5개 언어를 지원합니다: en, ko, es, pt, fr
        """
        if not self.is_loaded:
            self.load_model()
        
        # 언어 확인
        lang = request.language if request.language in self.SUPPORTED_LANGS else "en"
        if request.language and request.language not in self.SUPPORTED_LANGS:
            logger.warning(
                f"Language '{request.language}' not supported. "
                f"Supported: {self.SUPPORTED_LANGS}. Using 'en' instead."
            )
        
        # Voice style 매핑
        voice_name = self._map_voice_to_style(request.voice)
        logger.info(f"Using voice style: {voice_name}, language: {lang}")
        
        try:
            # Voice style 가져오기
            voice_style = self._get_voice_style(voice_name)
            
            # 음성 합성 (내장 모듈 사용 - lang 파라미터로 언어 지정)
            logger.debug(f"Synthesizing text ({len(request.text)} chars): {request.text[:50]}...")
            
            # 파라미터 추출 (extra_params 또는 기본값)
            extra = request.extra_params or {}
            
            # 속도 조정 (Supertonic 공식 기본값: 1.05, higher = faster)
            speed = request.speed if request.speed else 1.05
            
            # Denoising steps (기본값: 5, 높을수록 품질↑ 속도↓)
            total_steps = extra.get("total_steps", 5)
            
            # 청크 사이 무음 시간 (기본값: 0.3초)
            silence_duration = extra.get("silence_duration", 0.3)
            
            logger.debug(f"Params: speed={speed}, total_steps={total_steps}, silence_duration={silence_duration}")
            
            # ONNX 직접 추론 (lang 파라미터 사용)
            wav, duration = self.tts(
                text=request.text,
                lang=lang,
                style=voice_style,
                total_step=total_steps,
                speed=speed,
                silence_duration=silence_duration,
            )
            
            # duration이 배열인 경우 스칼라로 변환
            if isinstance(duration, np.ndarray):
                duration = float(duration.item() if duration.size == 1 else duration[0])
            
            logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
            
            # numpy 배열 형식 확인 및 변환
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            
            # 1D로 변환 (배치 차원 제거)
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            # float32로 변환
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # 정규화 (-1.0 ~ 1.0)
            max_val = max(abs(wav.max()), abs(wav.min()))
            if max_val > 1.0:
                wav = wav / max_val
            
            return TTSOutput(
                audio=wav,
                sample_rate=self._sample_rate,
                metadata={
                    "model": self.model_id,
                    "language": lang,
                    "voice_style": voice_name,
                    "speed": speed,
                    "duration": duration,
                    "engine": "supertonic-onnx"
                }
            )
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Synthesis failed: {e}")
            logger.error(f"Traceback:\n{error_traceback}")
            raise RuntimeError(f"Supertonic synthesis failed: {e}") from e
    
    def _map_voice_to_style(self, voice: str) -> str:
        """Voice ID를 Supertonic voice style로 매핑합니다."""
        if not voice:
            return "M1"
            
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
        """기본 샘플링 레이트 (모델 로드 후 실제 값 사용)"""
        return self._sample_rate if self._sample_rate else 24000
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False  # Supertonic-2는 배치 모드만 지원
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return False  # Supertonic-2는 사전 정의된 voice styles만 지원
