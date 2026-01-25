"""KaniTTS Engine

NineNineSix의 고속 고품질 TTS 모델.
LLM + Neural Audio Codec 조합으로 실시간 대화형 AI에 최적화.

Reference:
- HuggingFace: https://huggingface.co/nineninesix/kani-tts-370m
- License: Apache 2.0

Specifications:
- Model Size: 370M parameters
- Sample Rate: 22kHz
- Languages: English, German, Chinese, Korean, Arabic, Spanish
- Latency: ~1s for 15s audio (RTX 5080)
- Memory: 2GB GPU VRAM

Speakers:
- david (English British), puck (English Gemini), kore (English Gemini)
- andrew (English), jenny (English Irish), simon (English), katie (English)
- seulgi (Korean)
- bert (German), thorsten (German Hessisch)
- maria (Spanish)
- mei (Chinese Cantonese), ming (Chinese Shanghai)
- karim (Arabic), nur (Arabic)
"""

from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class KaniTTSEngine(BaseTTSEngine):
    """KaniTTS Engine
    
    High-speed, high-fidelity TTS optimized for real-time conversational AI.
    
    Features:
    - Multi-speaker: 15+ voices in 6 languages
    - Low latency: ~1s for 15s audio
    - Low memory: 2GB GPU VRAM
    - Controllable: temperature, top_p, repetition_penalty
    """
    
    # 지원 언어
    SUPPORTED_LANGUAGES = ["en", "de", "zh", "ko", "ar", "es"]
    
    # 스피커별 언어 매핑
    SPEAKER_LANG = {
        "david": "en", "puck": "en", "kore": "en", "andrew": "en",
        "jenny": "en", "simon": "en", "katie": "en",
        "seulgi": "ko",
        "bert": "de", "thorsten": "de",
        "maria": "es",
        "mei": "zh", "ming": "zh",
        "karim": "ar", "nur": "ar",
    }
    
    # 언어별 기본 스피커
    DEFAULT_SPEAKERS = {
        "en": "david",
        "ko": "seulgi",
        "de": "bert",
        "zh": "mei",
        "ar": "karim",
        "es": "maria",
    }
    
    # 언어 코드 매핑
    LANG_MAP = {
        "english": "en", "german": "de", "chinese": "zh",
        "korean": "ko", "arabic": "ar", "spanish": "es",
    }
    
    def __init__(
        self,
        model_id: str = "nineninesix/kani-tts-370m",
        device: str = "auto",
        **kwargs
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID
            device: cuda, cpu, auto
        """
        super().__init__(model_id, device=device, **kwargs)
        self._sample_rate = 22000  # KaniTTS uses 22kHz
        self._supported_languages = self.SUPPORTED_LANGUAGES
        
        self.model = None
        
    def _ensure_kanitts_installed(self) -> None:
        """kani-tts 패키지가 설치되어 있는지 확인하고, 없으면 설치합니다."""
        try:
            import kani_tts
        except ImportError:
            logger.info("kani-tts not found, installing...")
            import subprocess
            import sys
            
            # kani-tts --no-deps로 설치 (nemo-toolkit 버전 충돌 우회)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "kani-tts", "--no-deps", "-q"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install kani-tts: {result.stderr}")
                raise ImportError(
                    "Failed to install kani-tts. "
                    "Try manually: pip install 'vtts[kanitts]'"
                )
            
            logger.info("kani-tts installed successfully")
    
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading KaniTTS model: {self.model_id}")
        
        # kani-tts 패키지 확인 및 자동 설치
        self._ensure_kanitts_installed()
        
        try:
            from kani_tts import KaniTTS
            
            # 디바이스 설정
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 로드
            self.model = KaniTTS(
                self.model_id,
                suppress_logs=True,
                show_info=False
            )
            
            self._sample_rate = self.model.sample_rate
            self.is_loaded = True
            
            # 스피커 목록 확인
            if hasattr(self.model, 'speaker_list') and self.model.speaker_list:
                logger.info(f"Available speakers: {self.model.speaker_list}")
            
            logger.info(f"Successfully loaded KaniTTS: {self.model_id}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Sample rate: {self._sample_rate} Hz")
            
        except ImportError as e:
            logger.error(f"KaniTTS import failed: {e}")
            logger.error("Install with: pip install kani-tts")
            raise ImportError(
                "kani-tts package not found. Install with: pip install kani-tts"
            )
        except Exception as e:
            logger.error(f"Failed to load KaniTTS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model: {self.model_id}")
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성을 합성합니다.
        
        Args:
            request: TTSRequest
            
        Returns:
            TTSOutput: 합성된 오디오
        """
        if not self.is_loaded:
            self.load_model()
        
        logger.info(f"Synthesizing with KaniTTS")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # 언어 매핑
            language = self._map_language(request.language or "en")
            
            # 스피커 결정
            speaker_id = self._get_speaker(request.voice, language)
            logger.info(f"Using speaker: {speaker_id}, language: {language}")
            
            # extra_params에서 추가 파라미터 가져오기
            temperature = 1.0
            top_p = 0.95
            max_new_tokens = 1200
            repetition_penalty = 1.1
            
            if request.extra_params:
                temperature = request.extra_params.get("temperature", 1.0)
                top_p = request.extra_params.get("top_p", 0.95)
                max_new_tokens = request.extra_params.get("max_new_tokens", 1200)
                repetition_penalty = request.extra_params.get("repetition_penalty", 1.1)
            
            # TTS 생성
            audio, generated_text = self.model(
                request.text,
                speaker_id=speaker_id
            )
            
            # numpy array로 변환
            if isinstance(audio, torch.Tensor):
                audio_data = audio.squeeze().cpu().numpy()
            else:
                audio_data = np.array(audio).squeeze()
            
            # float32 변환 및 정규화
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 정규화
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            duration = len(audio_data) / self._sample_rate
            logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
            
            return TTSOutput(
                audio=audio_data,
                sample_rate=self._sample_rate,
                metadata={
                    "model": self.model_id,
                    "speaker": speaker_id,
                    "language": language,
                    "duration": duration,
                    "engine": "kanitts"
                }
            )
            
        except Exception as e:
            logger.error(f"KaniTTS synthesis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"KaniTTS synthesis failed: {e}") from e
    
    def _map_language(self, lang: str) -> str:
        """언어 코드를 매핑합니다."""
        lang_lower = lang.lower()
        
        # 이미 올바른 코드인 경우
        if lang_lower in self.SUPPORTED_LANGUAGES:
            return lang_lower
        
        # 전체 이름에서 코드로 변환
        return self.LANG_MAP.get(lang_lower, "en")
    
    def _get_speaker(self, voice: str, language: str) -> str:
        """언어와 voice에 따라 스피커를 결정합니다."""
        voice_lower = voice.lower() if voice else ""
        
        # 직접 스피커 이름이 지정된 경우
        if voice_lower in self.SPEAKER_LANG:
            return voice_lower
        
        # 언어별 기본 스피커
        return self.DEFAULT_SPEAKERS.get(language, "david")
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        return list(self.SPEAKER_LANG.keys())
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return False  # KaniTTS는 고정 스피커 사용
