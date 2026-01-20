"""Server State Management"""

from typing import Optional
import torch
from fastapi import FastAPI
from loguru import logger

from vtts.engines.registry import EngineRegistry
from vtts.engines.base import BaseTTSEngine
from vtts.engines.stt_base import BaseSTTEngine


class ServerState:
    """서버 전역 상태"""
    
    @staticmethod
    def initialize(
        app: FastAPI,
        model_id: str,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        stt_model_id: Optional[str] = None
    ):
        """서버 상태를 초기화합니다."""
        
        # 디바이스 자동 선택
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-selected device: {device}")
        
        # TTS 엔진 초기화
        engine_class = EngineRegistry.get_engine_for_model(model_id)
        if engine_class is None:
            raise ValueError(f"No TTS engine found for model: {model_id}")
        
        engine: BaseTTSEngine = engine_class(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir
        )
        engine.load_model()
        
        # STT 엔진 초기화 (선택적)
        stt_engine = None
        if stt_model_id:
            try:
                from vtts.engines.faster_whisper import FasterWhisperEngine
                stt_engine: BaseSTTEngine = FasterWhisperEngine(
                    model_id=stt_model_id,
                    device=device,
                    cache_dir=cache_dir
                )
                stt_engine.load_model()
                logger.info(f"STT engine loaded: {stt_model_id}")
            except Exception as e:
                logger.warning(f"Failed to load STT engine: {e}")
        
        # 앱 상태에 저장
        app.state.model_id = model_id
        app.state.device = device
        app.state.cache_dir = cache_dir
        app.state.engine = engine
        app.state.stt_engine = stt_engine
        app.state.stt_model_id = stt_model_id
        
        logger.info(f"Server state initialized with TTS model: {model_id}")
