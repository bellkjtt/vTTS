"""GPT-SoVITS TTS Engine - Real Implementation"""

from pathlib import Path
from typing import List, Optional, Union
import os

import numpy as np
import torch
from loguru import logger

from vtts.engines.base import BaseTTSEngine, TTSOutput, TTSRequest


class GPTSoVITSEngine(BaseTTSEngine):
    """GPT-SoVITS TTS Engine
    
    Few-shot and zero-shot voice cloning from RVC-Boss.
    Supports Chinese, English, Japanese, Korean, Cantonese.
    
    Features:
    - Few-shot: 1 minute training data
    - Zero-shot: 5 second reference audio
    - Cross-lingual synthesis
    
    Supported languages: zh, en, ja, ko, yue (Cantonese)
    """
    
    SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "yue"]
    
    def __init__(
        self,
        model_id: str = "kevinwang676/GPT-SoVITS-v3",
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self._supported_languages = self.SUPPORTED_LANGUAGES
        self._sample_rate = 32000  # GPT-SoVITS uses 32kHz
        self.tts_pipeline = None
        
    def load_model(self) -> None:
        """모델을 로드합니다."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} is already loaded")
            return
        
        logger.info(f"Loading GPT-SoVITS model: {self.model_id}")
        logger.info("This requires GPT-SoVITS package to be installed")
        
        try:
            # GPT-SoVITS 임포트
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            
            # HuggingFace에서 모델 다운로드
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model from HuggingFace: {self.model_id}")
            model_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Model downloaded to: {model_dir}")
            
            # TTS 설정
            # GPT-SoVITS는 SoVITS와 GPT 모델 경로를 각각 지정
            sovits_path = os.path.join(model_dir, "SoVITS_weights", "model.pth")
            gpt_path = os.path.join(model_dir, "GPT_weights", "model.ckpt")
            
            # TTS Config 생성
            tts_config = TTS_Config()
            tts_config.t2s_weights_path = gpt_path
            tts_config.vits_weights_path = sovits_path
            tts_config.device = self.device
            tts_config.is_half = self.device == "cuda"  # FP16 for GPU
            
            # TTS 초기화
            self.tts_pipeline = TTS(tts_config)
            self.is_loaded = True
            
            logger.info(f"Successfully loaded GPT-SoVITS model: {self.model_id}")
            logger.info("GPT-SoVITS supports few-shot and zero-shot voice cloning")
            
        except ImportError as e:
            logger.error(
                "GPT-SoVITS package not installed. "
                "Install from: https://github.com/RVC-Boss/GPT-SoVITS"
            )
            raise ImportError(
                "GPT-SoVITS package required. "
                "Clone and install from: git clone https://github.com/RVC-Boss/GPT-SoVITS.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load GPT-SoVITS model: {e}")
            logger.error(f"Model directory: {model_dir if 'model_dir' in locals() else 'N/A'}")
            raise
    
    def unload_model(self) -> None:
        """모델을 언로드합니다."""
        if self.tts_pipeline is not None:
            del self.tts_pipeline
            self.tts_pipeline = None
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
                f"Language '{request.language}' may not be supported. "
                f"Supported: {self._supported_languages}"
            )
        
        # 참조 오디오 필수 확인
        if request.reference_audio is None:
            raise ValueError(
                "GPT-SoVITS requires reference audio for voice cloning. "
                "Please provide reference_audio parameter."
            )
        
        if request.reference_text is None:
            raise ValueError(
                "GPT-SoVITS requires reference text (what is said in reference audio). "
                "Please provide reference_text parameter."
            )
        
        try:
            # 참조 오디오 로드
            ref_audio_path = self._prepare_reference_audio(request.reference_audio)
            
            logger.info(f"Using reference audio: {ref_audio_path}")
            logger.info(f"Reference text: {request.reference_text}")
            logger.info(f"Synthesizing: {request.text[:50]}...")
            
            # GPT-SoVITS inference
            # get_tts_wav 함수 호출
            synthesis_result = self.tts_pipeline.get_tts_wav(
                ref_wav_path=ref_audio_path,
                prompt_text=request.reference_text,
                prompt_language=self._map_language_code(request.language),
                text=request.text,
                text_language=self._map_language_code(request.language),
                how_to_cut="凑四句一切",  # 문장 분할 방식
                top_k=20,
                top_p=0.6,
                temperature=0.6,
                ref_free=False  # reference-free 모드 비활성화
            )
            
            # 결과 처리
            # synthesis_result는 (sample_rate, audio_data) 튜플
            if isinstance(synthesis_result, tuple):
                sr, audio_data = synthesis_result
            else:
                audio_data = synthesis_result
                sr = self._sample_rate
            
            # numpy 배열로 변환
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            elif not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # 1D 배열로 변환
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            
            # float32 변환 및 정규화
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
            
            duration = len(audio_data) / sr
            logger.info(f"Synthesis complete: {duration:.2f}s audio generated")
            
            return TTSOutput(
                audio=audio_data,
                sample_rate=sr,
                metadata={
                    "model": self.model_id,
                    "language": request.language,
                    "reference_audio": str(ref_audio_path),
                    "reference_text": request.reference_text,
                    "duration": duration,
                    "engine": "gpt-sovits"
                }
            )
            
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis failed: {e}")
            raise RuntimeError(f"GPT-SoVITS synthesis failed: {e}") from e
    
    def _prepare_reference_audio(
        self,
        reference: Union[str, Path, np.ndarray]
    ) -> str:
        """참조 오디오를 준비합니다."""
        import tempfile
        import soundfile as sf
        
        if isinstance(reference, (str, Path)):
            # 이미 파일 경로면 그대로 반환
            return str(reference)
        
        elif isinstance(reference, np.ndarray):
            # numpy 배열이면 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
                dir=tempfile.gettempdir()
            )
            
            # WAV 파일로 저장
            sf.write(temp_file.name, reference, self._sample_rate)
            logger.debug(f"Saved reference audio to: {temp_file.name}")
            return temp_file.name
        
        else:
            raise ValueError(f"Unsupported reference audio type: {type(reference)}")
    
    def _map_language_code(self, lang: str) -> str:
        """언어 코드를 GPT-SoVITS 형식으로 매핑"""
        # GPT-SoVITS는 "中文", "英文", "日文", "粤语", "韩文" 형식 사용
        mapping = {
            "zh": "中文",
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "yue": "粤语",
            "chinese": "中文",
            "english": "英文",
            "japanese": "日文",
            "korean": "韩文",
            "cantonese": "粤语",
        }
        
        lang_lower = lang.lower()
        return mapping.get(lang_lower, "中文")
    
    @property
    def supported_languages(self) -> List[str]:
        """지원하는 언어"""
        return self._supported_languages
    
    @property
    def supported_voices(self) -> List[str]:
        """지원하는 음성"""
        # GPT-SoVITS는 reference audio로 음성을 결정
        return ["reference"]
    
    @property
    def default_sample_rate(self) -> int:
        """기본 샘플링 레이트"""
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        """스트리밍 지원 여부"""
        return False  # GPT-SoVITS는 배치 모드
    
    @property
    def supports_zero_shot(self) -> bool:
        """Zero-shot 지원 여부"""
        return True  # GPT-SoVITS는 zero-shot 지원 (5초 참조 오디오)
