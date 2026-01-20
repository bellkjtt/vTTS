"""Pydantic Models for API"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """TTS 요청 (OpenAI 호환)"""
    model: str = Field(..., description="모델 ID")
    input: str = Field(..., description="합성할 텍스트")
    voice: str = Field(default="default", description="음성 ID")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="응답 오디오 포맷"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="음성 속도")
    
    # vTTS 확장 필드
    language: Optional[str] = Field(default="ko", description="언어 코드")
    reference_audio: Optional[str] = Field(default=None, description="참조 오디오 (base64 또는 URL)")
    reference_text: Optional[str] = Field(default=None, description="참조 텍스트")


class ModelInfo(BaseModel):
    """모델 정보"""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "vtts"


class ModelListResponse(BaseModel):
    """모델 목록 응답"""
    object: str = "list"
    data: List[ModelInfo]


class VoiceInfo(BaseModel):
    """음성 정보"""
    id: str
    name: str
    language: str


class VoiceListResponse(BaseModel):
    """음성 목록 응답"""
    voices: List[VoiceInfo]


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model: str
    device: str
    is_loaded: bool
