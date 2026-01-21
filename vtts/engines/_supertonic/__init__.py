"""
Supertonic 2.0.0 ONNX 직접 추론 모듈

다국어 지원: en, ko, es, pt, fr

GitHub 소스 기반:
https://github.com/supertone-inc/supertonic/tree/main/py
"""

from .helper import (
    AVAILABLE_LANGS,
    UnicodeProcessor,
    TextToSpeech,
    Style,
    load_text_to_speech,
    load_voice_style,
    load_cfgs,
)

__all__ = [
    "AVAILABLE_LANGS",
    "UnicodeProcessor", 
    "TextToSpeech",
    "Style",
    "load_text_to_speech",
    "load_voice_style",
    "load_cfgs",
]

__version__ = "2.0.0"
