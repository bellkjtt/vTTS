"""Faster-Whisper STT Engine"""

from pathlib import Path
from typing import List, Optional, Union
import io

import numpy as np
import torch
from loguru import logger

from vtts.engines.stt_base import BaseSTTEngine, STTOutput, STTRequest


class FasterWhisperEngine(BaseSTTEngine):
    """Faster-Whisper STT Engine
    
    High-performance Whisper implementation using CTranslate2.
    """
    
    def __init__(
        self,
        model_id: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        **kwargs
    ):
        """
        Args:
            model_id: Whisper 모델 크기 (tiny, base, small, medium, large-v2, large-v3)
                     또는 Huggingface 모델 ID
            device: 디바이스 (cuda, cpu, auto)
            compute_type: 컴퓨팅 타입 (int8, float16, float32, auto)
        """
        super().__init__(model_id, device, **kwargs)
        
        # 디바이스 자동 선택
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 컴퓨팅 타입 자동 선택
        if compute_type == "auto":
            if self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
        
        self.compute_type = compute_type
        
        # 지원 언어 (Whisper는 99개 언어 지원)
        self._supported_languages = [
            "en", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "ru",
            "ar", "tr", "vi", "th", "id", "ms", "hi", "bn", "ta", "te"
            # ... 더 많은 언어
        ]
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading Faster-Whisper model: {self.model_id}")
        
        try:
            from faster_whisper import WhisperModel
            
            # 모델 로드
            self.model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded Faster-Whisper model: {self.model_id}")
            
        except ImportError:
            logger.error(
                "Faster-Whisper not installed. "
                "Install with: pip install faster-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {self.model_id}")
    
    def transcribe(self, request: STTRequest) -> STTOutput:
        """음성을 텍스트로 변환합니다."""
        if not self.is_loaded:
            self.load_model()
        
        # 오디오 로드
        audio = self._load_audio(request.audio)
        
        # Faster-Whisper 옵션
        transcribe_options = {
            "language": request.language,
            "task": request.task,
            "temperature": request.temperature,
        }
        
        # 타임스탬프 옵션
        word_timestamps = False
        if request.timestamp_granularities:
            word_timestamps = "word" in request.timestamp_granularities
        
        # 전사 실행
        segments, info = self.model.transcribe(
            audio,
            word_timestamps=word_timestamps,
            **transcribe_options
        )
        
        # 결과 수집
        all_text = []
        all_segments = []
        
        for segment in segments:
            all_text.append(segment.text)
            
            segment_dict = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            
            # 단어 타임스탬프
            if word_timestamps and hasattr(segment, 'words'):
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]
            
            all_segments.append(segment_dict)
        
        full_text = " ".join(all_text).strip()
        
        return STTOutput(
            text=full_text,
            language=info.language,
            segments=all_segments if request.response_format in ["json", "verbose_json"] else None,
            metadata={
                "model": self.model_id,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None,
            }
        )
    
    def _load_audio(self, audio: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """오디오를 numpy 배열로 로드합니다."""
        if isinstance(audio, np.ndarray):
            return audio
        
        if isinstance(audio, bytes):
            # 바이트에서 로드
            import soundfile as sf
            audio_array, sr = sf.read(io.BytesIO(audio))
            
            # 16kHz로 리샘플링 (Whisper 요구사항)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
            return audio_array
        
        # 파일 경로에서 로드
        import soundfile as sf
        audio_array, sr = sf.read(str(audio))
        
        # 16kHz로 리샘플링
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
        return audio_array
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supports_translation(self) -> bool:
        """번역 지원 여부"""
        return True  # Whisper는 모든 언어 -> 영어 번역 지원
    
    @property
    def supports_timestamps(self) -> bool:
        """타임스탬프 지원 여부"""
        return True
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False  # Faster-Whisper는 배치 모드만 지원
