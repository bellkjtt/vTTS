"""STT API Routes (OpenAI Whisper Compatible)"""

import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from vtts.engines.stt_base import STTRequest as EngineSTTRequest

stt_router = APIRouter(prefix="/v1/audio", tags=["stt"])


@stt_router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """음성을 텍스트로 변환 (OpenAI Whisper API 호환)
    
    Examples:
        curl http://localhost:8000/v1/audio/transcriptions \\
          -H "Content-Type: multipart/form-data" \\
          -F file="@audio.mp3" \\
          -F model="large-v3" \\
          -F language="ko"
    """
    stt_engine = request.app.state.stt_engine
    
    if stt_engine is None:
        raise HTTPException(
            status_code=400,
            detail="No STT engine loaded. Start server with STT model."
        )
    
    try:
        # 오디오 파일 읽기
        audio_bytes = await file.read()
        
        # 타임스탬프 파싱
        granularities = None
        if timestamp_granularities:
            granularities = [g.strip() for g in timestamp_granularities.split(",")]
        
        # 엔진 요청 생성
        engine_request = EngineSTTRequest(
            audio=audio_bytes,
            language=language,
            task="transcribe",
            temperature=temperature,
            timestamp_granularities=granularities,
            response_format=response_format
        )
        
        # 전사 실행
        logger.info(f"Transcribing audio file: {file.filename}")
        start_time = time.time()
        
        output = stt_engine.transcribe(engine_request)
        
        elapsed = time.time() - start_time
        logger.info(f"Transcription complete: {elapsed:.2f}s")
        
        # 응답 포맷에 따라 반환
        if response_format == "text":
            return JSONResponse(
                content=output.text,
                media_type="text/plain"
            )
        
        elif response_format == "json":
            return {
                "text": output.text
            }
        
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": output.language,
                "duration": output.metadata.get("duration") if output.metadata else None,
                "text": output.text,
                "segments": output.segments,
            }
        
        elif response_format == "srt":
            srt_content = _format_srt(output.segments)
            return JSONResponse(
                content=srt_content,
                media_type="text/plain"
            )
        
        elif response_format == "vtt":
            vtt_content = _format_vtt(output.segments)
            return JSONResponse(
                content=vtt_content,
                media_type="text/plain"
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response format: {response_format}"
            )
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@stt_router.post("/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """음성을 영어로 번역 (OpenAI Whisper API 호환)
    
    Examples:
        curl http://localhost:8000/v1/audio/translations \\
          -H "Content-Type: multipart/form-data" \\
          -F file="@audio.mp3" \\
          -F model="large-v3"
    """
    stt_engine = request.app.state.stt_engine
    
    if stt_engine is None:
        raise HTTPException(
            status_code=400,
            detail="No STT engine loaded. Start server with STT model."
        )
    
    if not stt_engine.supports_translation:
        raise HTTPException(
            status_code=400,
            detail="This model does not support translation"
        )
    
    try:
        # 오디오 파일 읽기
        audio_bytes = await file.read()
        
        # 엔진 요청 생성
        engine_request = EngineSTTRequest(
            audio=audio_bytes,
            language=None,  # 자동 감지
            task="translate",  # 번역 모드
            temperature=temperature,
            response_format=response_format
        )
        
        # 번역 실행
        logger.info(f"Translating audio file: {file.filename}")
        start_time = time.time()
        
        output = stt_engine.transcribe(engine_request)
        
        elapsed = time.time() - start_time
        logger.info(f"Translation complete: {elapsed:.2f}s")
        
        # 응답 포맷에 따라 반환
        if response_format == "text":
            return JSONResponse(
                content=output.text,
                media_type="text/plain"
            )
        else:
            return {
                "text": output.text
            }
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _format_srt(segments: list) -> str:
    """SRT 포맷으로 변환"""
    if not segments:
        return ""
    
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = _format_timestamp_srt(segment["start"])
        end = _format_timestamp_srt(segment["end"])
        text = segment["text"].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def _format_vtt(segments: list) -> str:
    """WebVTT 포맷으로 변환"""
    if not segments:
        return "WEBVTT\n\n"
    
    vtt_lines = ["WEBVTT", ""]
    
    for i, segment in enumerate(segments, 1):
        start = _format_timestamp_vtt(segment["start"])
        end = _format_timestamp_vtt(segment["end"])
        text = segment["text"].strip()
        
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(text)
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)


def _format_timestamp_srt(seconds: float) -> str:
    """SRT 타임스탬프 포맷 (00:00:00,000)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """VTT 타임스탬프 포맷 (00:00:00.000)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
