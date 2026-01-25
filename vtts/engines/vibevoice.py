"""VibeVoice TTS Engine for vTTS

Microsoft VibeVoice - Realtime Streaming TTS
~300ms first audible latency, streaming text input

Reference: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
License: MIT
"""

from typing import Optional, Generator
import numpy as np
import torch
from loguru import logger

from .base import BaseTTSEngine, TTSOutput, TTSRequest


class VibeVoiceEngine(BaseTTSEngine):
    """VibeVoice Realtime TTS Engine
    
    Microsoft VibeVoice - Realtime streaming TTS
    
    특징:
    - ~300ms first audible latency
    - Streaming text input 지원
    - 영어 전용 (고품질)
    - transformers 호환
    
    License: MIT
    """
    
    SUPPORTED_LANGUAGES = ["en"]
    
    def __init__(
        self,
        model_id: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.sample_rate = 24000  # VibeVoice default
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
            
        logger.info(f"Loading VibeVoice model: {self.model_id}")
        logger.info("This may take a while on first run (~2GB download)")
        
        try:
            from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
            
            # Tokenizer 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Feature extractor 로드
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 모델 로드
            self.model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Device로 이동
            self.model = self.model.to(self._device)
            
            # Sample rate 가져오기
            if hasattr(self.feature_extractor, 'sampling_rate'):
                self.sample_rate = self.feature_extractor.sampling_rate
            
            self.is_loaded = True
            logger.info(f"Successfully loaded VibeVoice model: {self.model_id}")
            logger.info(f"Device: {self._device}")
            logger.info(f"Sample rate: {self.sample_rate} Hz")
            
        except Exception as e:
            logger.error(f"Failed to load VibeVoice model: {e}")
            raise
            
    def unload_model(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("VibeVoice model unloaded")
        
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성 합성"""
        if not self.is_loaded:
            self.load_model()
            
        logger.info(f"Synthesizing with VibeVoice")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # 입력 토큰화
            inputs = self.tokenizer(
                request.text,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=2048
                )
            
            # numpy로 변환
            if hasattr(output, 'audio'):
                audio = output.audio.cpu().numpy().squeeze()
            else:
                audio = output.cpu().numpy().squeeze()
            
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
            logger.error(f"VibeVoice synthesis failed: {e}")
            raise RuntimeError(f"VibeVoice synthesis failed: {e}") from e
            
    def synthesize_stream(self, request: TTSRequest) -> Generator[TTSOutput, None, None]:
        """스트리밍 음성 합성 (VibeVoice 특화 기능)"""
        if not self.is_loaded:
            self.load_model()
            
        logger.info(f"Streaming synthesis with VibeVoice")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # 스트리밍 생성
            inputs = self.tokenizer(
                request.text,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # VibeVoice는 streaming generation을 지원
            with torch.no_grad():
                for chunk in self.model.generate_stream(
                    **inputs,
                    chunk_size=512
                ):
                    if hasattr(chunk, 'audio'):
                        audio = chunk.audio.cpu().numpy().squeeze()
                    else:
                        audio = chunk.cpu().numpy().squeeze()
                    
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    yield TTSOutput(
                        audio=audio,
                        sample_rate=self.sample_rate
                    )
                    
        except AttributeError:
            # generate_stream이 없으면 일반 생성 후 청크로 분할
            logger.warning("Streaming not available, falling back to chunked output")
            output = self.synthesize(request)
            yield output
        except Exception as e:
            logger.error(f"VibeVoice streaming failed: {e}")
            raise RuntimeError(f"VibeVoice streaming failed: {e}") from e
            
    def get_available_voices(self) -> list:
        """사용 가능한 음성 목록 (단일 화자)"""
        return ["default"]
        
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
        return True  # VibeVoice Realtime은 스트리밍 지원
        
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 음성 복제 지원 여부"""
        return False  # 단일 화자


# 엔진 등록
def register():
    from .registry import EngineRegistry
    EngineRegistry.register("vibevoice", VibeVoiceEngine, ["microsoft/VibeVoice*", "*vibevoice*"])
