"""GPT-SoVITS v3/v4 TTS Engine

Zero-shot 및 Few-shot 음성 클로닝을 지원하는 고품질 TTS 엔진.

Reference:
- GitHub: https://github.com/RVC-Boss/GPT-SoVITS
- HuggingFace: https://huggingface.co/kevinwang676/GPT-SoVITS-v3

Supported Languages: zh (Chinese), en (English), ja (Japanese), ko (Korean), yue (Cantonese)

Requirements:
- GPT-SoVITS 저장소 클론 필요
- pip install vtts[gptsovits]
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Generator

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class GPTSoVITSEngine(BaseTTSEngine):
    """GPT-SoVITS v3/v4 TTS Engine
    
    Few-shot and zero-shot voice cloning from RVC-Boss.
    
    Features:
    - Few-shot: 1 minute training data로 고품질 음성 클로닝
    - Zero-shot: 5 second reference audio로 음성 클로닝
    - Cross-lingual synthesis (다른 언어로 합성)
    - v3: 24kHz, 고품질
    - v4: 48kHz, 최고품질
    
    Supported languages: zh, en, ja, ko, yue (Cantonese)
    
    Note: reference_audio 파라미터 필수!
    """
    
    SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "yue"]
    
    # 언어 코드 매핑 (GPT-SoVITS 내부 형식)
    LANG_MAP = {
        "zh": "zh",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "yue": "yue",
        "chinese": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "cantonese": "yue",
    }
    
    def __init__(
        self,
        model_id: str = "kevinwang676/GPT-SoVITS-v3",
        version: str = "v3",
        device: str = "auto",
        **kwargs
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID 또는 로컬 경로
            version: GPT-SoVITS 버전 (v1, v2, v3, v4)
            device: cuda, cpu, auto
        """
        super().__init__(model_id, device=device, **kwargs)
        self._supported_languages = self.SUPPORTED_LANGUAGES
        self._version = version
        
        # 버전별 샘플레이트
        self._sample_rate = 48000 if version == "v4" else 32000
        
        self.tts_pipeline = None
        self.tts_config = None
        self._gpt_sovits_path = None
        
    def _setup_gpt_sovits_path(self) -> str:
        """GPT-SoVITS 경로를 설정합니다."""
        # 1. 환경변수 확인
        gpt_sovits_path = os.environ.get("GPT_SOVITS_PATH")
        
        if gpt_sovits_path and os.path.exists(gpt_sovits_path):
            return gpt_sovits_path
        
        # 2. 가능한 경로들 확인
        possible_paths = [
            # vtts setup으로 설치된 경로 (우선)
            Path.home() / ".vtts" / "GPT-SoVITS",
            # 사용자 홈 디렉토리
            Path.home() / "GPT-SoVITS",
            # 시스템 경로
            Path("/opt/GPT-SoVITS"),
            # 현재 디렉토리
            Path("GPT-SoVITS"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "GPT-SoVITS not found. Please run:\n\n"
            "  vtts setup --engine gptsovits\n\n"
            "Or manually:\n"
            "  git clone https://github.com/RVC-Boss/GPT-SoVITS.git ~/.vtts/GPT-SoVITS\n"
            "  cd ~/.vtts/GPT-SoVITS && pip install -r requirements.txt\n"
            "  export GPT_SOVITS_PATH=~/.vtts/GPT-SoVITS"
        )
    
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading GPT-SoVITS model: {self.model_id}")
        logger.info(f"Version: {self._version}")
        
        try:
            # GPT-SoVITS 경로 설정
            self._gpt_sovits_path = self._setup_gpt_sovits_path()
            logger.info(f"GPT-SoVITS path: {self._gpt_sovits_path}")
            
            # HuggingFace에서 pretrained 모델 다운로드
            logger.info(f"Downloading pretrained models from HuggingFace: {self.model_id}")
            model_path = snapshot_download(
                repo_id=self.model_id,
                allow_patterns=["GPT_SoVITS/pretrained_models/**"],  # pretrained_models만 다운로드
                cache_dir=None,  # 기본 HF 캐시 사용 (~/.cache/huggingface/)
                resume_download=True
            )
            logger.info(f"Pretrained models downloaded to: {model_path}")
            
            # Pretrained 모델 경로 설정
            pretrained_dir = os.path.join(model_path, "GPT_SoVITS", "pretrained_models")
            
            # sys.path에 추가
            if self._gpt_sovits_path not in sys.path:
                sys.path.insert(0, self._gpt_sovits_path)
            
            gpt_sovits_module = os.path.join(self._gpt_sovits_path, "GPT_SoVITS")
            if gpt_sovits_module not in sys.path:
                sys.path.insert(0, gpt_sovits_module)
            
            # 현재 디렉토리 변경 (GPT-SoVITS가 상대 경로 사용)
            original_cwd = os.getcwd()
            os.chdir(self._gpt_sovits_path)
            
            try:
                # GPT-SoVITS 임포트
                from TTS_infer_pack.TTS import TTS, TTS_Config
                
                # 설정 파일 경로
                config_path = os.path.join(
                    self._gpt_sovits_path, 
                    "GPT_SoVITS", "configs", "tts_infer.yaml"
                )
                
                # TTS_Config 초기화
                self.tts_config = TTS_Config(config_path)
                
                # 디바이스 설정
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # 버전별 설정
                self.tts_config.device = self.device
                self.tts_config.is_half = self.device == "cuda"
                
                # 버전별 모델 경로 설정 (로컬 GPT-SoVITS 저장소 기준)
                if self._version == "v3":
                    self.tts_config.t2s_weights_path = os.path.join(
                        pretrained_dir, "s1v3.ckpt"
                    )
                    self.tts_config.vits_weights_path = os.path.join(
                        pretrained_dir, "s2Gv3.pth"
                    )
                    # BERT 및 CNHubert 경로 설정
                    self.tts_config.bert_base_path = os.path.join(
                        pretrained_dir, "chinese-roberta-wwm-ext-large"
                    )
                    self.tts_config.cnhuhbert_base_path = os.path.join(
                        pretrained_dir, "chinese-hubert-base"
                    )
                elif self._version == "v4":
                    self.tts_config.t2s_weights_path = os.path.join(
                        pretrained_dir, "s1v3.ckpt"
                    )
                    self.tts_config.vits_weights_path = os.path.join(
                        pretrained_dir, "s2Gv4.pth"
                    )
                elif self._version == "v2":
                    self.tts_config.t2s_weights_path = os.path.join(
                        pretrained_dir, "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
                    )
                    self.tts_config.vits_weights_path = os.path.join(
                        pretrained_dir, "gsv-v2final-pretrained/s2G2333k.pth"
                    )
                
                # TTS 파이프라인 초기화
                logger.info("Initializing TTS pipeline...")
                self.tts_pipeline = TTS(self.tts_config)
                
                self.is_loaded = True
                logger.info(f"Successfully loaded GPT-SoVITS {self._version}")
                logger.info(f"Device: {self.device}")
                logger.info(f"Sample rate: {self._sample_rate} Hz")
                
            finally:
                # 원래 디렉토리로 복귀
                os.chdir(original_cwd)
                
        except ImportError as e:
            logger.error(f"GPT-SoVITS import failed: {e}")
            raise ImportError(
                "GPT-SoVITS 패키지를 찾을 수 없습니다.\n"
                "1. git clone https://github.com/RVC-Boss/GPT-SoVITS.git\n"
                "2. cd GPT-SoVITS && pip install -r requirements.txt\n"
                "3. export GPT_SOVITS_PATH=/path/to/GPT-SoVITS"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load GPT-SoVITS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.tts_pipeline is not None:
            del self.tts_pipeline
            self.tts_pipeline = None
        
        if self.tts_config is not None:
            del self.tts_config
            self.tts_config = None
        
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model: {self.model_id}")
    
    def synthesize(self, request: TTSRequest) -> TTSOutput:
        """음성을 합성합니다.
        
        GPT-SoVITS는 참조 오디오가 필수입니다!
        
        Args:
            request: TTSRequest (reference_audio 필수)
            
        Returns:
            TTSOutput: 합성된 오디오
            
        Raises:
            ValueError: reference_audio가 없는 경우
        """
        if not self.is_loaded:
            self.load_model()
        
        # 참조 오디오 필수 확인
        if not request.reference_audio:
            raise ValueError(
                "GPT-SoVITS requires reference_audio for voice cloning.\n"
                "Please provide reference_audio parameter with a path to audio file."
            )
        
        # 언어 매핑
        text_lang = self._map_language(request.language or "zh")
        prompt_lang = text_lang  # 기본적으로 같은 언어 사용
        
        logger.info(f"Synthesizing with GPT-SoVITS {self._version}")
        logger.info(f"Text: {request.text[:50]}...")
        logger.info(f"Language: {text_lang}")
        logger.info(f"Reference audio: {request.reference_audio}")
        
        try:
            # 원래 디렉토리 저장
            original_cwd = os.getcwd()
            os.chdir(self._gpt_sovits_path)
            
            try:
                # 참조 오디오 경로 준비
                ref_audio_path = self._prepare_reference_audio(request.reference_audio)
                
                # 추론 요청 생성
                inference_req = {
                    "text": request.text,
                    "text_lang": text_lang,
                    "ref_audio_path": ref_audio_path,
                    "prompt_text": request.reference_text or "",
                    "prompt_lang": prompt_lang,
                    "top_k": 15,
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "speed_factor": request.speed if request.speed else 1.0,
                    "fragment_interval": 0.3,
                    "seed": -1,
                    "parallel_infer": True,
                    "repetition_penalty": 1.35,
                    "sample_steps": 32,  # v3 기본값
                    "streaming_mode": False,
                }
                
                # extra_params에서 추가 파라미터 가져오기
                if request.extra_params:
                    for key in ["top_k", "top_p", "temperature", "sample_steps", "seed"]:
                        if key in request.extra_params:
                            inference_req[key] = request.extra_params[key]
                
                # TTS 추론 실행
                tts_generator = self.tts_pipeline.run(inference_req)
                
                # 결과 수집
                audio_chunks = []
                sample_rate = self._sample_rate
                
                for sr, chunk in tts_generator:
                    sample_rate = sr
                    audio_chunks.append(chunk)
                
                # 청크 합치기
                if audio_chunks:
                    audio_data = np.concatenate(audio_chunks)
                else:
                    raise RuntimeError("No audio generated")
                
                # float32 변환 및 정규화
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 정규화
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val
                
                duration = len(audio_data) / sample_rate
                logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
                
                return TTSOutput(
                    audio=audio_data,
                    sample_rate=sample_rate,
                    metadata={
                        "model": self.model_id,
                        "version": self._version,
                        "language": text_lang,
                        "reference_audio": str(ref_audio_path),
                        "duration": duration,
                        "engine": "gpt-sovits"
                    }
                )
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"GPT-SoVITS synthesis failed: {e}") from e
    
    def _prepare_reference_audio(self, reference: Union[str, Path, bytes]) -> str:
        """참조 오디오를 준비합니다."""
        import soundfile as sf
        import base64
        
        if isinstance(reference, (str, Path)):
            path = Path(reference)
            if path.exists():
                return str(path.absolute())
            
            # base64 인코딩된 데이터인지 확인
            if reference.startswith("data:audio"):
                # data:audio/wav;base64,xxxxx 형식
                header, data = reference.split(",", 1)
                audio_bytes = base64.b64decode(data)
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                )
                temp_file.write(audio_bytes)
                temp_file.close()
                return temp_file.name
            
            raise FileNotFoundError(f"Reference audio not found: {reference}")
        
        elif isinstance(reference, bytes):
            # bytes 데이터
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            )
            temp_file.write(reference)
            temp_file.close()
            return temp_file.name
        
        else:
            raise ValueError(f"Unsupported reference audio type: {type(reference)}")
    
    def _map_language(self, lang: str) -> str:
        """언어 코드를 매핑합니다."""
        lang_lower = lang.lower()
        return self.LANG_MAP.get(lang_lower, "zh")
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성 (GPT-SoVITS는 reference audio 기반)"""
        return ["reference"]
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return True  # GPT-SoVITS v2+ 스트리밍 지원
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return True  # GPT-SoVITS는 zero-shot 지원
