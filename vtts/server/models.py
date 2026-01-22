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
    
    # Supertonic 엔진 파라미터
    total_steps: Optional[int] = Field(
        default=None, 
        ge=1, le=20,
        description="Denoising steps (기본값: 5, 높을수록 품질↑ 속도↓)"
    )
    silence_duration: Optional[float] = Field(
        default=None,
        ge=0.0, le=2.0,
        description="청크 사이 무음 시간 (초, 기본값: 0.3)"
    )
    
    # GPT-SoVITS 엔진 파라미터
    top_k: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description="Top-K 샘플링 (기본값: 15)"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Top-P 샘플링 (기본값: 1.0)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.1, le=2.0,
        description="생성 다양성 (기본값: 1.0, 높을수록 다양)"
    )
    sample_steps: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description="샘플링 스텝 수 (기본값: 32 for v3)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="랜덤 시드 (기본값: -1, 재현성 위해 설정)"
    )


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
