"""FastAPI Server Application"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from vtts.engines.registry import EngineRegistry
from vtts.server.routes import router
from vtts.server.stt_routes import stt_router
from vtts.server.state import ServerState


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # Startup
    logger.info("Starting vTTS server...")
    logger.info(f"Loaded model: {app.state.model_id}")
    logger.info(f"Engine: {app.state.engine.__class__.__name__}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down vTTS server...")
    if hasattr(app.state, 'engine') and app.state.engine is not None:
        app.state.engine.unload_model()


def create_app(
    model_id: str,
    device: str = "auto",
    cache_dir: Optional[str] = None,
    stt_model_id: Optional[str] = None
) -> FastAPI:
    """FastAPI 앱을 생성합니다.
    
    Args:
        model_id: Huggingface 모델 ID
        device: 디바이스 (cuda, cpu, auto)
        cache_dir: 모델 캐시 디렉토리
        
    Returns:
        FastAPI 앱
    """
    app = FastAPI(
        title="vTTS - Universal TTS API",
        description="vLLM for Text-to-Speech",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS 미들웨어
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 라우터 등록
    app.include_router(router)
    app.include_router(stt_router)
    
    # 상태 초기화
    ServerState.initialize(
        app=app,
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
        stt_model_id=stt_model_id
    )
    
    return app
