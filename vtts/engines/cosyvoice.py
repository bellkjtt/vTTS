"""CosyVoice3 TTS Engine - Embedded Implementation

CosyVoice 코드가 vtts/engines/_cosyvoice/에 내장되어 있어
별도의 클론이나 설치가 필요 없습니다.
"""

from pathlib import Path
from typing import List, Optional, Union
import os

import numpy as np
import torch
from loguru import logger
from huggingface_hub import snapshot_download

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class CosyVoiceEngine(BaseTTSEngine):
    """CosyVoice3 TTS Engine (Embedded)
    
    Zero-shot multilingual TTS from FunAudioLLM (Alibaba).
    Supports 9 languages and 18+ Chinese dialects.
    
    CosyVoice 코드가 vtts에 내장되어 있어 별도 설치가 필요 없습니다!
    모델만 HuggingFace에서 자동 다운로드됩니다.
    
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
        
        try:
            # sys.modules에 cosyvoice 및 matcha alias 등록
            # HuggingFace 모델의 YAML이 'cosyvoice.xxx' 경로를 사용하므로 필요
            import sys
            import importlib
            
            pkg_path = 'vtts.engines._cosyvoice'
            
            # _cosyvoice 모듈 동적 import (circular import 방지)
            _cosyvoice = importlib.import_module(pkg_path)
            _matcha = importlib.import_module(f'{pkg_path}.matcha')
            
            # 최상위 모듈 등록
            sys.modules['cosyvoice'] = _cosyvoice
            sys.modules['matcha'] = _matcha
            
            # matcha 하위 모듈 등록
            matcha_submodules = [
                'models', 'models.components', 'models.components.flow_matching',
                'models.components.decoder', 'models.components.transformer',
                'hifigan', 'hifigan.models', 'hifigan.xutils',
            ]
            for submod in matcha_submodules:
                try:
                    full_name = f'{pkg_path}.matcha.{submod}'
                    alias_name = f'matcha.{submod}'
                    mod = importlib.import_module(full_name)
                    sys.modules[alias_name] = mod
                except ImportError as e:
                    logger.debug(f"Failed to import matcha.{submod}: {e}")
            
            # cosyvoice 하위 모듈 등록
            cosyvoice_submodules = [
                'cli', 'cli.cosyvoice', 'cli.frontend', 'cli.model',
                'llm', 'llm.llm',
                'flow', 'flow.flow', 'flow.flow_matching',
                'hifigan', 'hifigan.generator',
                'transformer', 'transformer.unet',
                'tokenizer', 'tokenizer.tokenizer',
                'utils', 'utils.common', 'utils.file_utils',
                'dataset', 'dataset.processor', 'dataset.dataset',
            ]
            
            for submod in cosyvoice_submodules:
                try:
                    full_name = f'{pkg_path}.{submod}'
                    alias_name = f'cosyvoice.{submod}'
                    mod = importlib.import_module(full_name)
                    sys.modules[alias_name] = mod
                    logger.debug(f"Registered: {alias_name}")
                except ImportError as e:
                    logger.warning(f"Failed to import {alias_name}: {e}")
            
            # 내장된 CosyVoice 모듈 import
            AutoModel = importlib.import_module(f'{pkg_path}.cli.cosyvoice').AutoModel
            
            # HuggingFace에서 모델 다운로드
            logger.info(f"Downloading model from HuggingFace: {self.model_id}")
            
            model_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Model downloaded to: {model_dir}")
            
            # CosyVoice 자동 초기화 (v1/v2/v3 자동 감지)
            self.model = AutoModel(model_dir=model_dir)
            self._sample_rate = self.model.sample_rate
            self.is_loaded = True
            
            logger.info(f"Successfully loaded CosyVoice model: {self.model_id}")
            logger.info(f"Sample rate: {self._sample_rate} Hz")
            logger.info(f"Model type: {self.model.__class__.__name__}")
            logger.info("CosyVoice supports zero-shot voice cloning")
            
            # 사용 가능한 speaker 목록
            available_spks = self.model.list_available_spks()
            if available_spks:
                logger.info(f"Available speakers: {available_spks[:5]}...")  # 처음 5개만
            
        except ImportError as e:
            logger.error(f"CosyVoice import failed: {e}")
            logger.error("Required dependencies may be missing.")
            logger.error("Install with: pip install vtts[cosyvoice]")
            raise
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
            # CosyVoice3는 zero-shot만 지원 (참조 오디오 필수)
            model_class = self.model.__class__.__name__
            
            if request.reference_audio is not None:
                # Zero-shot mode (참조 오디오 사용)
                output_audio = self._synthesize_zero_shot(request)
            elif model_class == 'CosyVoice3':
                # CosyVoice3는 SFT 모드 없음 - 기본 프롬프트로 zero-shot 사용
                logger.info("CosyVoice3 requires reference audio. Using default prompt.")
                output_audio = self._synthesize_zero_shot_default(request)
            else:
                # CosyVoice v1/v2: SFT mode (사전 학습된 음성 사용)
                output_audio = self._synthesize_sft(request)
            
            # numpy 배열로 변환
            if isinstance(output_audio, torch.Tensor):
                output_audio = output_audio.cpu().numpy()
            
            # flatten
            output_audio = output_audio.flatten()
            
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
                    "engine": "cosyvoice"
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
        
        # 속도 설정
        speed = request.speed if hasattr(request, 'speed') and request.speed else 1.0
        
        # inference_sft 호출
        audio_chunks = []
        for output in self.model.inference_sft(
            tts_text=request.text,
            spk_id=speaker,
            stream=False,
            speed=speed
        ):
            # output은 dict with 'tts_speech' key
            if isinstance(output, dict) and 'tts_speech' in output:
                audio = output['tts_speech']
                if isinstance(audio, torch.Tensor):
                    audio_chunks.append(audio.cpu().numpy())
                else:
                    audio_chunks.append(np.array(audio))
        
        if not audio_chunks:
            raise RuntimeError("No audio output from CosyVoice SFT mode")
        
        # 모든 청크 연결
        return np.concatenate([chunk.flatten() for chunk in audio_chunks])
    
    def _synthesize_zero_shot(self, request: TTSRequest) -> np.ndarray:
        """Zero-shot mode로 합성 (참조 오디오 사용)"""
        logger.info("Using CosyVoice zero-shot mode with reference audio")
        
        # 참조 오디오 로드
        reference_audio = self._load_reference_audio(request.reference_audio)
        
        # CosyVoice3는 특수한 프롬프트 형식 필요
        # 형식: "You are a helpful assistant.<|endofprompt|>참조 텍스트"
        model_class = self.model.__class__.__name__
        reference_text = request.reference_text or "참조 음성입니다."
        
        if model_class == 'CosyVoice3' and '<|endofprompt|>' not in reference_text:
            reference_text = f"You are a helpful assistant.<|endofprompt|>{reference_text}"
        
        # 속도 설정
        speed = request.speed if hasattr(request, 'speed') and request.speed else 1.0
        
        # inference_zero_shot 호출
        audio_chunks = []
        for output in self.model.inference_zero_shot(
            tts_text=request.text,
            prompt_text=reference_text,
            prompt_wav=reference_audio,
            stream=False,
            speed=speed
        ):
            if isinstance(output, dict) and 'tts_speech' in output:
                audio = output['tts_speech']
                if isinstance(audio, torch.Tensor):
                    audio_chunks.append(audio.cpu().numpy())
                else:
                    audio_chunks.append(np.array(audio))
        
        if not audio_chunks:
            raise RuntimeError("No audio output from CosyVoice zero-shot mode")
        
        # 모든 청크 연결
        return np.concatenate([chunk.flatten() for chunk in audio_chunks])
    
    def _synthesize_zero_shot_default(self, request: TTSRequest) -> np.ndarray:
        """CosyVoice3용 기본 zero-shot 합성 (기본 프롬프트 사용)"""
        logger.info("Using CosyVoice3 zero-shot mode with default prompt")
        
        # 기본 프롬프트 텍스트 (CosyVoice3 공식 예시)
        default_prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
        
        # 기본 프롬프트 오디오 생성 (모델 디렉토리에서 찾거나 생성)
        import os
        from huggingface_hub import hf_hub_download
        
        # 먼저 모델 디렉토리에서 기본 프롬프트 오디오 찾기
        model_dir = os.path.dirname(self.model.model_dir) if hasattr(self.model, 'model_dir') else None
        prompt_wav_path = None
        
        # HuggingFace에서 기본 프롬프트 오디오 다운로드 시도
        try:
            prompt_wav_path = hf_hub_download(
                repo_id="FunAudioLLM/CosyVoice-300M",
                filename="asset/zero_shot_prompt.wav",
                cache_dir=self.cache_dir
            )
            logger.info(f"Downloaded default prompt audio: {prompt_wav_path}")
        except Exception as e:
            logger.warning(f"Failed to download default prompt audio: {e}")
            # 프롬프트 오디오 없이는 CosyVoice3 사용 불가
            raise RuntimeError(
                "CosyVoice3 requires reference audio for synthesis. "
                "Please provide 'reference_audio' and 'reference_text' in your request."
            )
        
        # 속도 설정
        speed = request.speed if hasattr(request, 'speed') and request.speed else 1.0
        
        # inference_zero_shot 호출
        audio_chunks = []
        for output in self.model.inference_zero_shot(
            tts_text=request.text,
            prompt_text=default_prompt_text,
            prompt_wav=prompt_wav_path,
            stream=False,
            speed=speed
        ):
            if isinstance(output, dict) and 'tts_speech' in output:
                audio = output['tts_speech']
                if isinstance(audio, torch.Tensor):
                    audio_chunks.append(audio.cpu().numpy())
                else:
                    audio_chunks.append(np.array(audio))
        
        if not audio_chunks:
            raise RuntimeError("No audio output from CosyVoice3 zero-shot mode")
        
        # 모든 청크 연결
        return np.concatenate([chunk.flatten() for chunk in audio_chunks])
    
    def _load_reference_audio(self, reference: Union[str, Path, bytes, np.ndarray]) -> str:
        """참조 오디오를 로드합니다.
        
        CosyVoice는 파일 경로를 직접 받습니다.
        bytes나 ndarray인 경우 임시 파일로 저장합니다.
        """
        import tempfile
        import soundfile as sf
        
        if isinstance(reference, (str, Path)):
            path = str(reference)
            if os.path.exists(path):
                return path
            # base64인 경우
            import base64
            try:
                audio_bytes = base64.b64decode(reference)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_bytes)
                    return f.name
            except:
                raise FileNotFoundError(f"Reference audio not found: {path}")
        
        elif isinstance(reference, bytes):
            # bytes를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(reference)
                return f.name
        
        elif isinstance(reference, np.ndarray):
            # numpy array를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, reference, 16000)
                return f.name
        
        else:
            raise ValueError(f"Unsupported reference audio type: {type(reference)}")
    
    def _map_voice_to_speaker(self, voice: str) -> str:
        """Voice ID를 CosyVoice speaker ID로 매핑"""
        available = self.model.list_available_spks()
        logger.debug(f"Available speakers: {available}")
        
        # 사용 가능한 스피커가 없으면 None 반환 (zero-shot 모드 사용)
        if not available:
            logger.warning("No preset speakers available, will use zero-shot mode")
            return None
        
        # voice가 없으면 첫 번째 사용 가능한 speaker 사용
        if not voice:
            return available[0]
        
        # 먼저 사용 가능한 speakers에서 직접 찾기
        if voice in available:
            return voice
        
        voice_lower = voice.lower()
        
        # 기본 매핑
        mapping = {
            "default": available[0] if available else None,
            "male": "中文男",
            "female": "中文女",
            "chinese_male": "中文男",
            "chinese_female": "中文女",
            "english_male": "英文男",
            "english_female": "英文女",
            "japanese_male": "日语男",
            "japanese_female": "日语女",
            "korean_male": "韩语男",
            "korean_female": "韩语女",
            "f1": available[0] if available else None,
            "m1": available[0] if available else None,
        }
        
        # 매핑에서 찾기
        speaker = mapping.get(voice_lower)
        
        # 해당 speaker가 사용 가능한지 확인
        if speaker and speaker in available:
            return speaker
        
        # 없으면 첫 번째 사용 가능한 speaker 사용
        logger.warning(f"Speaker '{voice}' not available, using '{available[0]}'")
        return available[0]
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        base_voices = [
            "default",
            "chinese_male", "chinese_female",
            "english_male", "english_female",
            "japanese_male", "japanese_female",
            "korean_male", "korean_female"
        ]
        # 모델이 로드된 경우 실제 사용 가능한 speakers 추가
        if self.is_loaded and self.model:
            try:
                available = self.model.list_available_spks()
                return list(set(base_voices + available))
            except:
                pass
        return base_voices
    
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
