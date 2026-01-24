"""Chatterbox TTS Engine

Resemble AI의 State-of-the-Art 오픈소스 TTS 모델.
Zero-shot voice cloning, 23개 언어 지원, Emotion control.

Reference:
- GitHub: https://github.com/resemble-ai/chatterbox
- HuggingFace: https://huggingface.co/ResembleAI/chatterbox
- License: MIT

Models:
- Chatterbox: English, CFG & Exaggeration control (500M)
- Chatterbox-Multilingual: 23 languages (500M)
- Chatterbox-Turbo: English, Low latency, Paralinguistic tags (350M)

Supported Languages (Multilingual):
ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class ChatterboxEngine(BaseTTSEngine):
    """Chatterbox TTS Engine
    
    Zero-shot voice cloning TTS from Resemble AI.
    
    Features:
    - Zero-shot: 10초 reference audio로 음성 클로닝
    - Multilingual: 23개 언어 지원
    - Turbo: 저지연, Paralinguistic tags 지원
    - Emotion control: exaggeration, cfg_weight 조절
    
    Model Types:
    - english: Chatterbox (English only, CFG control)
    - multilingual: Chatterbox-Multilingual (23 languages)
    - turbo: Chatterbox-Turbo (Low latency, paralinguistic tags)
    """
    
    # 지원 언어 (Multilingual)
    SUPPORTED_LANGUAGES = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh"
    ]
    
    # 언어 코드 매핑
    LANG_MAP = {
        "arabic": "ar", "danish": "da", "german": "de", "greek": "el",
        "english": "en", "spanish": "es", "finnish": "fi", "french": "fr",
        "hebrew": "he", "hindi": "hi", "italian": "it", "japanese": "ja",
        "korean": "ko", "malay": "ms", "dutch": "nl", "norwegian": "no",
        "polish": "pl", "portuguese": "pt", "russian": "ru", "swedish": "sv",
        "swahili": "sw", "turkish": "tr", "chinese": "zh",
    }
    
    def __init__(
        self,
        model_id: str = "ResembleAI/chatterbox",
        model_type: str = "auto",  # auto, english, multilingual, turbo
        device: str = "auto",
        **kwargs
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID
            model_type: 모델 타입 (auto, english, multilingual, turbo)
            device: cuda, cpu, auto
        """
        super().__init__(model_id, device=device, **kwargs)
        
        # 모델 타입 자동 감지
        if model_type == "auto":
            model_id_lower = model_id.lower()
            if "turbo" in model_id_lower:
                self._model_type = "turbo"
            elif "multilingual" in model_id_lower or "mtl" in model_id_lower:
                self._model_type = "multilingual"
            else:
                self._model_type = "english"
        else:
            self._model_type = model_type
        
        self._sample_rate = 24000  # Chatterbox uses 24kHz
        self._supported_languages = ["en"] if self._model_type == "english" else self.SUPPORTED_LANGUAGES
        
        self.model = None
        
    def _ensure_chatterbox_installed(self) -> None:
        """chatterbox-tts 패키지가 설치되어 있는지 확인하고, 없으면 --no-deps로 설치합니다."""
        try:
            import chatterbox
        except ImportError:
            logger.info("chatterbox-tts not found, installing with --no-deps...")
            import subprocess
            import sys
            
            # --no-deps로 설치하여 transformers 버전 충돌 우회
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "chatterbox-tts", "--no-deps", "-q"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install chatterbox-tts: {result.stderr}")
                raise ImportError(
                    "Failed to install chatterbox-tts. "
                    "Try manually: pip install chatterbox-tts --no-deps"
                )
            
            logger.info("chatterbox-tts installed successfully (--no-deps)")
    
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading Chatterbox model: {self.model_id}")
        logger.info(f"Model type: {self._model_type}")
        
        # chatterbox-tts 패키지 확인 및 자동 설치
        self._ensure_chatterbox_installed()
        
        try:
            # 디바이스 설정
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self._model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                logger.info("Loaded Chatterbox-Turbo model")
                
            elif self._model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                logger.info("Loaded Chatterbox-Multilingual model")
                
            else:  # english
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                logger.info("Loaded Chatterbox (English) model")
            
            self._sample_rate = self.model.sr
            self.is_loaded = True
            
            logger.info(f"Successfully loaded Chatterbox {self._model_type}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Sample rate: {self._sample_rate} Hz")
            
        except ImportError as e:
            logger.error(f"Chatterbox import failed: {e}")
            logger.error("Install with: pip install chatterbox-tts --no-deps")
            raise ImportError(
                "chatterbox-tts package not found. Install with: pip install chatterbox-tts --no-deps"
            )
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
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
        
        logger.info(f"Synthesizing with Chatterbox {self._model_type}")
        logger.info(f"Text: {request.text[:50]}...")
        
        try:
            # 참조 오디오 준비
            audio_prompt_path = None
            if request.reference_audio:
                audio_prompt_path = self._prepare_reference_audio(request.reference_audio)
                logger.info(f"Reference audio: {audio_prompt_path}")
            
            # 언어 매핑
            language = self._map_language(request.language or "en")
            
            # extra_params에서 추가 파라미터 가져오기
            exaggeration = 0.5
            cfg_weight = 0.5
            
            if request.extra_params:
                exaggeration = request.extra_params.get("exaggeration", 0.5)
                cfg_weight = request.extra_params.get("cfg_weight", 0.5)
            
            # TTS 생성
            if self._model_type == "turbo":
                # Turbo는 exaggeration/cfg 없음, audio_prompt 필수
                if audio_prompt_path:
                    wav = self.model.generate(
                        request.text,
                        audio_prompt_path=audio_prompt_path
                    )
                else:
                    wav = self.model.generate(request.text)
                    
            elif self._model_type == "multilingual":
                # Multilingual
                if audio_prompt_path:
                    wav = self.model.generate(
                        request.text,
                        language_id=language,
                        audio_prompt_path=audio_prompt_path
                    )
                else:
                    wav = self.model.generate(
                        request.text,
                        language_id=language
                    )
                    
            else:  # english
                # English with CFG control
                if audio_prompt_path:
                    wav = self.model.generate(
                        request.text,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
                else:
                    wav = self.model.generate(
                        request.text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
            
            # Tensor to numpy
            if isinstance(wav, torch.Tensor):
                audio_data = wav.squeeze().cpu().numpy()
            else:
                audio_data = np.array(wav).squeeze()
            
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
                    "model_type": self._model_type,
                    "language": language,
                    "duration": duration,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "engine": "chatterbox"
                }
            )
            
        except Exception as e:
            logger.error(f"Chatterbox synthesis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Chatterbox synthesis failed: {e}") from e
    
    def _prepare_reference_audio(self, reference: Union[str, Path, bytes]) -> str:
        """참조 오디오를 준비합니다."""
        import base64
        import soundfile as sf
        
        if isinstance(reference, (str, Path)):
            ref_str = str(reference)
            
            # base64 데이터 감지
            if ref_str.startswith("data:audio"):
                header, data = ref_str.split(",", 1)
                audio_bytes = base64.b64decode(data)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(audio_bytes)
                temp_file.close()
                return temp_file.name
            
            # raw base64 감지
            if len(ref_str) > 100 and not os.path.sep in ref_str[:50]:
                try:
                    audio_bytes = base64.b64decode(ref_str)
                    if audio_bytes[:4] in [b'RIFF', b'ID3\x03', b'OggS', b'fLaC'] or audio_bytes[:2] == b'\xff\xfb':
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_file.write(audio_bytes)
                        temp_file.close()
                        return temp_file.name
                except Exception:
                    pass
            
            # 파일 경로
            path = Path(ref_str)
            if not path.is_absolute():
                path = path.resolve()
            
            if path.exists():
                return str(path)
            
            raise FileNotFoundError(f"Reference audio not found: {reference}")
        
        elif isinstance(reference, bytes):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(reference)
            temp_file.close()
            return temp_file.name
        
        else:
            raise ValueError(f"Unsupported reference audio type: {type(reference)}")
    
    def _map_language(self, lang: str) -> str:
        """언어 코드를 매핑합니다."""
        lang_lower = lang.lower()
        
        # 이미 올바른 코드인 경우
        if lang_lower in self.SUPPORTED_LANGUAGES:
            return lang_lower
        
        # 전체 이름에서 코드로 변환
        return self.LANG_MAP.get(lang_lower, "en")
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성 (Chatterbox는 reference audio 기반)"""
        return ["reference", "default"]
    
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
        return True
