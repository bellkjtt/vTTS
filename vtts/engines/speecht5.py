"""SpeechT5 TTS Engine for vTTS

Microsoft SpeechT5 - transformers 기반 TTS
speaker embedding으로 다양한 음성 지원

Reference: https://huggingface.co/microsoft/speecht5_tts
License: MIT
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger

from .base import BaseTTSEngine, TTSOutput, TTSRequest


class SpeechT5Engine(BaseTTSEngine):
    """SpeechT5 TTS Engine
    
    Microsoft SpeechT5 - encoder-decoder TTS
    
    특징:
    - Speaker embedding 지원 (7,306개 화자)
    - 고품질 영어 음성
    - transformers 완벽 호환
    
    License: MIT
    """
    
    SUPPORTED_LANGUAGES = ["en"]
    
    # CMU ARCTIC 화자 ID
    SPEAKER_IDS = {
        "default": 0,
        "male1": 0,
        "female1": 1000,
        "male2": 2000,
        "female2": 3000,
        "male3": 4000,
        "female3": 5000,
    }
    
    def __init__(
        self,
        model_id: str = "microsoft/speecht5_tts",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.sample_rate = 16000  # SpeechT5 default
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
            
        logger.info(f"Loading SpeechT5 model: {self.model_id}")
        logger.info("This may take a while on first run...")
        
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            from datasets import load_dataset
            
            # Processor 로드
            self.processor = SpeechT5Processor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # 모델 로드
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Vocoder 로드
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Speaker embeddings 로드
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors",
                split="validation",
                cache_dir=self.cache_dir
            )
            self.speaker_embeddings = torch.tensor(
                embeddings_dataset[0]["xvector"]
            ).unsqueeze(0)
            
            # 전체 화자 임베딩 저장 (선택적 사용)
            self._all_embeddings = embeddings_dataset
            
            # Device로 이동
            self.model = self.model.to(self._device)
            self.vocoder = self.vocoder.to(self._device)
            self.speaker_embeddings = self.speaker_embeddings.to(self._device)
            
            self.is_loaded = True
            logger.info(f"Successfully loaded SpeechT5 model: {self.model_id}")
            logger.info(f"Device: {self._device}")
            logger.info(f"Sample rate: {self.sample_rate} Hz")
            logger.info(f"Available speakers: {len(self._all_embeddings)}")
            
        except Exception as e:
            logger.error(f"Failed to load SpeechT5 model: {e}")
            raise
            
    def unload_model(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.vocoder is not None:
            del self.vocoder
            self.vocoder = None
        if self.speaker_embeddings is not None:
            del self.speaker_embeddings
            self.speaker_embeddings = None
        self._all_embeddings = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("SpeechT5 model unloaded")
        
    def _get_speaker_embedding(self, voice: Optional[str]) -> torch.Tensor:
        """Speaker embedding 결정"""
        if voice is None:
            return self.speaker_embeddings
            
        # 숫자로 직접 지정한 경우
        if voice.isdigit():
            idx = int(voice) % len(self._all_embeddings)
            return torch.tensor(
                self._all_embeddings[idx]["xvector"]
            ).unsqueeze(0).to(self._device)
            
        # 프리셋 이름인 경우
        if voice.lower() in self.SPEAKER_IDS:
            idx = self.SPEAKER_IDS[voice.lower()] % len(self._all_embeddings)
            return torch.tensor(
                self._all_embeddings[idx]["xvector"]
            ).unsqueeze(0).to(self._device)
            
        # 기본값
        return self.speaker_embeddings
        
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성 합성"""
        if not self.is_loaded:
            self.load_model()
            
        logger.info(f"Synthesizing with SpeechT5")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # Speaker embedding 결정
            speaker_embedding = self._get_speaker_embedding(request.voice)
            
            # 입력 처리
            inputs = self.processor(text=request.text, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_embedding,
                    vocoder=self.vocoder
                )
            
            # numpy로 변환
            audio = speech.cpu().numpy()
            
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
            logger.error(f"SpeechT5 synthesis failed: {e}")
            raise RuntimeError(f"SpeechT5 synthesis failed: {e}") from e
            
    def get_available_voices(self) -> list:
        """사용 가능한 음성 프리셋 목록"""
        return list(self.SPEAKER_IDS.keys())
        
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
        return False  # Speaker embedding 기반


# 엔진 등록
def register():
    from .registry import EngineRegistry
    EngineRegistry.register("speecht5", SpeechT5Engine, ["microsoft/speecht5*", "*speecht5*"])
