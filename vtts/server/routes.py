"""API Routes"""

import io
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from vtts.server.models import (
    TTSRequest,
    ModelInfo,
    ModelListResponse,
    VoiceInfo,
    VoiceListResponse,
    HealthResponse
)
from vtts.engines.base import TTSRequest as EngineTTSRequest
from vtts.utils.audio import encode_audio

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> HealthResponse:
    """헬스 체크"""
    engine = request.app.state.engine
    
    return HealthResponse(
        status="ok",
        model=request.app.state.model_id,
        device=request.app.state.device,
        is_loaded=engine.is_loaded
    )


@router.get("/v1/models")
async def list_models(request: Request) -> ModelListResponse:
    """사용 가능한 모델 목록 (OpenAI 호환)"""
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id=request.app.state.model_id,
                object="model",
                created=int(time.time()),
                owned_by="vtts"
            )
        ]
    )


@router.get("/v1/voices")
async def list_voices(request: Request) -> VoiceListResponse:
    """사용 가능한 음성 목록"""
    engine = request.app.state.engine
    
    voices = []
    for voice_id in engine.supported_voices:
        for lang in engine.supported_languages:
            voices.append(VoiceInfo(
                id=voice_id,
                name=voice_id,
                language=lang
            ))
    
    return VoiceListResponse(voices=voices)


@router.post("/v1/audio/speech")
async def create_speech(
    request: Request,
    tts_request: TTSRequest
) -> StreamingResponse:
    """음성 합성 (OpenAI 호환)
    
    Examples:
        curl http://localhost:8000/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "Supertone/supertonic-2",
            "input": "안녕하세요, vTTS입니다.",
            "voice": "default",
            "language": "ko"
          }' \\
          --output speech.mp3
    """
    engine = request.app.state.engine
    
    try:
        # 엔진 요청 변환
        engine_request = EngineTTSRequest(
            text=tts_request.input,
            language=tts_request.language or "ko",
            voice=tts_request.voice,
            speed=tts_request.speed,
            reference_audio=tts_request.reference_audio,
            reference_text=tts_request.reference_text,
            stream=False
        )
        
        # 음성 합성
        logger.info(f"Synthesizing: {tts_request.input[:50]}...")
        start_time = time.time()
        
        output = engine.synthesize(engine_request)
        
        elapsed = time.time() - start_time
        audio_duration = len(output.audio) / output.sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        
        logger.info(
            f"Synthesis complete: {elapsed:.2f}s, "
            f"audio: {audio_duration:.2f}s, RTF: {rtf:.3f}"
        )
        
        # 오디오 인코딩
        audio_bytes = encode_audio(
            output.audio,
            output.sample_rate,
            format=tts_request.response_format
        )
        
        # MIME 타입 매핑
        mime_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=mime_types.get(tts_request.response_format, "audio/mpeg"),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{tts_request.response_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "name": "vTTS",
        "version": "0.1.0",
        "description": "Universal TTS Serving System",
        "docs": "/docs"
    }
