"""TTS Engine Registry"""

from typing import Dict, Optional, Type
from loguru import logger

from vtts.engines.base import BaseTTSEngine


class EngineRegistry:
    """TTS 엔진 레지스트리"""
    
    _engines: Dict[str, Type[BaseTTSEngine]] = {}
    _model_patterns: Dict[str, str] = {}  # model pattern -> engine name
    
    @classmethod
    def register(cls, name: str, engine_class: Type[BaseTTSEngine], model_patterns: Optional[list[str]] = None):
        """
        엔진을 등록합니다.
        
        Args:
            name: 엔진 이름
            engine_class: 엔진 클래스
            model_patterns: 모델 ID 패턴 (예: ["Supertone/*", "supertonic*"])
        """
        cls._engines[name] = engine_class
        logger.info(f"Registered TTS engine: {name}")
        
        if model_patterns:
            for pattern in model_patterns:
                cls._model_patterns[pattern.lower()] = name
                logger.debug(f"Registered model pattern: {pattern} -> {name}")
    
    @classmethod
    def get_engine(cls, name: str) -> Optional[Type[BaseTTSEngine]]:
        """엔진을 가져옵니다."""
        return cls._engines.get(name)
    
    @classmethod
    def get_engine_for_model(cls, model_id: str) -> Optional[Type[BaseTTSEngine]]:
        """
        모델 ID에 맞는 엔진을 자동으로 찾습니다.
        
        Args:
            model_id: Huggingface 모델 ID
            
        Returns:
            엔진 클래스 또는 None
        """
        model_id_lower = model_id.lower()
        
        # 정확한 패턴 매칭
        for pattern, engine_name in cls._model_patterns.items():
            pattern_clean = pattern.replace("*", "").lower()
            
            # 양쪽에 * 있는 경우 (e.g., *gpt-sovits*)
            if pattern.startswith("*") and pattern.endswith("*"):
                if pattern_clean in model_id_lower:
                    return cls._engines.get(engine_name)
            # 뒤에만 * 있는 경우 (e.g., Supertone/*)
            elif pattern.endswith("*"):
                if model_id_lower.startswith(pattern_clean):
                    return cls._engines.get(engine_name)
            # 앞에만 * 있는 경우 (e.g., *supertonic)
            elif pattern.startswith("*"):
                if model_id_lower.endswith(pattern_clean):
                    return cls._engines.get(engine_name)
            else:
                if pattern_clean in model_id_lower:
                    return cls._engines.get(engine_name)
        
        logger.warning(f"No engine found for model: {model_id}")
        return None
    
    @classmethod
    def list_engines(cls) -> list[str]:
        """등록된 모든 엔진 목록을 반환합니다."""
        return list(cls._engines.keys())
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, list[str]]:
        """엔진별 지원 모델 패턴을 반환합니다."""
        result = {}
        for pattern, engine_name in cls._model_patterns.items():
            if engine_name not in result:
                result[engine_name] = []
            result[engine_name].append(pattern)
        return result


# 엔진 자동 임포트 및 등록
def auto_register_engines():
    """사용 가능한 모든 엔진을 자동으로 등록합니다."""
    try:
        from vtts.engines.supertonic import SupertonicEngine
        EngineRegistry.register(
            "supertonic",
            SupertonicEngine,
            model_patterns=["Supertone/*", "*supertonic*"]
        )
    except ImportError as e:
        logger.debug(f"Supertonic engine not available: {e}")
    
    try:
        from vtts.engines.cosyvoice import CosyVoiceEngine
        EngineRegistry.register(
            "cosyvoice",
            CosyVoiceEngine,
            model_patterns=["FunAudioLLM/*", "*cosyvoice*", "*CosyVoice*"]
        )
    except ImportError as e:
        logger.debug(f"CosyVoice engine not available: {e}")
    
    try:
        from vtts.engines.gptsovits import GPTSoVITSEngine
        EngineRegistry.register(
            "gptsovits",
            GPTSoVITSEngine,
            model_patterns=["kevinwang676/*", "lj1995/*", "*GPT-SoVITS*", "*gpt-sovits*", "*gptsovits*"]
        )
    except ImportError as e:
        logger.debug(f"GPT-SoVITS engine not available: {e}")
    
    try:
        from vtts.engines.qwen3tts import Qwen3TTSEngine
        EngineRegistry.register(
            "qwen3tts",
            Qwen3TTSEngine,
            model_patterns=["Qwen/Qwen3-TTS*", "*Qwen3-TTS*", "*qwen3-tts*"]
        )
    except ImportError as e:
        logger.debug(f"Qwen3-TTS engine not available: {e}")
    
    try:
        from vtts.engines.chatterbox import ChatterboxEngine
        EngineRegistry.register(
            "chatterbox",
            ChatterboxEngine,
            model_patterns=[
                "ResembleAI/*",
                "*chatterbox*",
                "*Chatterbox*"
            ]
        )
    except ImportError as e:
        logger.debug(f"Chatterbox engine not available: {e}")
    
    try:
        from vtts.engines.kanitts import KaniTTSEngine
        EngineRegistry.register(
            "kanitts",
            KaniTTSEngine,
            model_patterns=[
                "nineninesix/*",
                "*kani-tts*",
                "*KaniTTS*"
            ]
        )
    except ImportError as e:
        logger.debug(f"KaniTTS engine not available: {e}")
    
    try:
        from vtts.engines.bark import BarkEngine
        EngineRegistry.register(
            "bark",
            BarkEngine,
            model_patterns=[
                "suno/bark*",
                "*bark*"
            ]
        )
    except ImportError as e:
        logger.debug(f"Bark engine not available: {e}")
    
    logger.info(f"Auto-registered {len(EngineRegistry.list_engines())} TTS engines")


# 모듈 임포트 시 자동 등록
auto_register_engines()
