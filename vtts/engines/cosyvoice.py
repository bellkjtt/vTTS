"""CosyVoice3 TTS Engine - Real Implementation"""

from pathlib import Path
from typing import List, Optional, Union
import os

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class CosyVoiceEngine(BaseTTSEngine):
    """CosyVoice3 TTS Engine
    
    Zero-shot multilingual TTS from FunAudioLLM (Alibaba).
    Supports 9 languages and 18+ Chinese dialects.
    
    Supported languages: zh, en, ja, ko, yue, es, fr, de, pt
    Model sizes: 0.5B, 1.5B parameters
    """
    
    # Supported languages for CosyVoice3
    SUPPORTED_LANGUAGES = [
        "zh",  # Chinese (Mandarin)
        "en",  # English
        "ja",  # Japanese
        "ko",  # Korean
        "yue", # Cantonese
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "pt",  # Portuguese
    ]
    
    def __init__(
        self,
        model_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self._supported_languages = self.SUPPORTED_LANGUAGES
        self._sample_rate = 22050  # CosyVoice uses 22.05kHz
        self.model = None
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading CosyVoice model: {self.model_id}")
        logger.info("This requires CosyVoice package to be installed")
        
        try:
            # CosyVoice 임포트
            from cosyvoice.cli.cosyvoice import CosyVoice
            from cosyvoice.utils.file_utils import load_wav
            
            # 모델 로드
            # HuggingFace 모델 사용
            logger.info(f"Loading from HuggingFace: {self.model_id}")
            
            # CosyVoice는 model_dir을 직접 지정해야 함
            # HuggingFace에서 다운로드
            from huggingface_hub import snapshot_download
            
            model_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Model downloaded to: {model_dir}")
            
            # CosyVoice 초기화
            self.model = CosyVoice(model_dir)
            self.is_loaded = True
            
            logger.info(f"Successfully loaded CosyVoice model: {self.model_id}")
            logger.info("CosyVoice supports zero-shot voice cloning")
            
        except ImportError as e:
            logger.error(
                "CosyVoice package not installed. "
                "Install from: https://github.com/FunAudioLLM/CosyVoice"
            )
            raise ImportError(
                "CosyVoice package required. "
                "Clone and install from: git clone https://github.com/FunAudioLLM/CosyVoice.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load CosyVoice model: {e}")
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
        """음성을 합성합니다."""
        if not self.is_loaded:
            self.load_model()
        
        # 언어 확인
        if request.language not in self._supported_languages:
            logger.warning(
                f"Language '{request.language}' may not be officially supported. "
                f"Supported: {self._supported_languages}"
            )
        
        try:
            # CosyVoice inference mode 결정
            if request.reference_audio is not None:
                # Zero-shot mode (참조 오디오 사용)
                output_audio = self._synthesize_zero_shot(request)
            else:
                # SFT mode (사전 학습된 음성 사용)
                output_audio = self._synthesize_sft(request)
            
            # numpy 배열로 변환
            if isinstance(output_audio, torch.Tensor):
                output_audio = output_audio.cpu().numpy()
            
            # 정규화
            if output_audio.dtype != np.float32:
                output_audio = output_audio.astype(np.float32)
            
            if output_audio.max() > 1.0 or output_audio.min() < -1.0:
                output_audio = output_audio / max(abs(output_audio.max()), abs(output_audio.min()))
            
            duration = len(output_audio) / self._sample_rate
            logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
            
            return TTSOutput(
                audio=output_audio,
                sample_rate=self._sample_rate,
                metadata={
                    "model": self.model_id,
                    "language": request.language,
                    "mode": "zero-shot" if request.reference_audio else "sft",
                    "duration": duration,
                    "engine": "cosyvoice3"
                }
            )
            
        except Exception as e:
            logger.error(f"CosyVoice synthesis failed: {e}")
            raise RuntimeError(f"CosyVoice synthesis failed: {e}") from e
    
    def _synthesize_sft(self, request: TTSRequest) -> np.ndarray:
        """SFT (Supervised Fine-Tuning) mode로 합성"""
        logger.info("Using CosyVoice SFT mode with preset voice")
        
        # CosyVoice SFT는 사전 정의된 speaker를 사용
        # voice를 speaker ID로 매핑
        speaker = self._map_voice_to_speaker(request.voice)
        
        # inference_sft 호출
        for output in self.model.inference_sft(
            tts_text=request.text,
            spk_id=speaker,
            stream=False
        ):
            # output은 dict with 'tts_speech' key
            if isinstance(output, dict) and 'tts_speech' in output:
                audio = output['tts_speech']
                if isinstance(audio, torch.Tensor):
                    return audio.cpu().numpy().flatten()
                return np.array(audio).flatten()
        
        raise RuntimeError("No audio output from CosyVoice SFT mode")
    
    def _synthesize_zero_shot(self, request: TTSRequest) -> np.ndarray:
        """Zero-shot mode로 합성 (참조 오디오 사용)"""
        logger.info("Using CosyVoice zero-shot mode with reference audio")
        
        # 참조 오디오 로드
        reference_audio = self._load_reference_audio(request.reference_audio)
        reference_text = request.reference_text or "참조 음성입니다."
        
        # inference_zero_shot 호출
        for output in self.model.inference_zero_shot(
            tts_text=request.text,
            prompt_text=reference_text,
            prompt_speech_16k=reference_audio,
            stream=False
        ):
            if isinstance(output, dict) and 'tts_speech' in output:
                audio = output['tts_speech']
                if isinstance(audio, torch.Tensor):
                    return audio.cpu().numpy().flatten()
                return np.array(audio).flatten()
        
        raise RuntimeError("No audio output from CosyVoice zero-shot mode")
    
    def _load_reference_audio(self, reference: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """참조 오디오를 로드합니다 (16kHz 필요)"""
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        
        if isinstance(reference, (str, Path)):
            # 파일에서 로드
            wav, sr = torchaudio.load(str(reference))
            
            # 16kHz로 리샘플
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            return wav.mean(dim=0)  # mono
        
        elif isinstance(reference, np.ndarray):
            # numpy array를 tensor로 변환
            wav = torch.from_numpy(reference).float()
            if wav.ndim > 1:
                wav = wav.mean(dim=0)
            return wav
        
        else:
            raise ValueError(f"Unsupported reference audio type: {type(reference)}")
    
    def _map_voice_to_speaker(self, voice: str) -> str:
        """Voice ID를 CosyVoice speaker ID로 매핑"""
        # CosyVoice는 여러 speaker를 지원
        # 기본 speakers: 중국어_남성, 중국어_여성, 영어_남성, 영어_여성 등
        
        voice_lower = voice.lower()
        
        # 기본 매핑
        mapping = {
            "default": "中文女",
            "male": "中文男",
            "female": "中문女",
            "chinese_male": "中文男",
            "chinese_female": "中文女",
            "english_male": "英文男",
            "english_female": "英文女",
            "japanese_male": "日语男",
            "japanese_female": "日语女",
            "korean_male": "韩语男",
            "korean_female": "韩语女",
        }
        
        return mapping.get(voice_lower, "中文女")
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        return [
            "default",
            "chinese_male", "chinese_female",
            "english_male", "english_female",
            "japanese_male", "japanese_female",
            "korean_male", "korean_female"
        ]
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return True  # CosyVoice3는 스트리밍 지원
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return True  # CosyVoice3는 zero-shot 지원
