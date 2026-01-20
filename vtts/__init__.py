"""vTTS - Universal TTS Serving System"""

__version__ = "0.1.0-beta"

from vtts.client import VTTSClient
from vtts.engines.registry import EngineRegistry

__all__ = ["VTTSClient", "EngineRegistry"]
