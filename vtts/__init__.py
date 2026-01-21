"""vTTS - Universal TTS Serving System

vLLM for Text-to-Speech
"""

__version__ = "0.1.0-beta"

import warnings

# ============================================================
# 환경 호환성 자동 확인
# ============================================================
def _check_environment():
    """Import 시 환경 호환성을 확인합니다."""
    
    # numpy 버전 확인 (2.0 이상은 호환성 문제)
    try:
        import numpy as np
        np_version = np.__version__
        major = int(np_version.split('.')[0])
        
        if major >= 2:
            warnings.warn(
                f"vTTS: numpy {np_version}이(가) 감지되었습니다. "
                f"numpy 2.x는 onnxruntime과 호환성 문제가 있을 수 있습니다. "
                f"문제 발생 시: vtts doctor --fix",
                UserWarning,
                stacklevel=2
            )
    except ImportError:
        pass
    
    # onnxruntime 확인
    try:
        import onnxruntime as ort
        # C 헤더 호환성 테스트 (문제 있으면 여기서 에러 발생)
        _ = ort.get_available_providers()
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            warnings.warn(
                "vTTS: numpy와 onnxruntime 간 바이너리 호환성 문제가 감지되었습니다. "
                "해결: vtts doctor --fix",
                UserWarning,
                stacklevel=2
            )
    except ImportError:
        pass

# 환경 확인 실행
_check_environment()

# ============================================================
# Public API
# ============================================================
from vtts.client import VTTSClient
from vtts.engines.registry import EngineRegistry

__all__ = ["VTTSClient", "EngineRegistry", "__version__"]
