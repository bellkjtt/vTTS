"""CosyVoice TTS Engine - Embedded Implementation

자동으로 matcha 및 cosyvoice 모듈 alias를 등록합니다.
"""
import sys

# matcha alias 등록 (이 모듈이 import될 때 자동 실행)
from vtts.engines._cosyvoice import matcha as _matcha

# matcha 최상위 모듈 등록
if 'matcha' not in sys.modules:
    sys.modules['matcha'] = _matcha

# matcha 하위 모듈 등록
_matcha_submodules = {
    'matcha.models': 'vtts.engines._cosyvoice.matcha.models',
    'matcha.models.components': 'vtts.engines._cosyvoice.matcha.models.components',
    'matcha.models.components.flow_matching': 'vtts.engines._cosyvoice.matcha.models.components.flow_matching',
}

for alias, full_name in _matcha_submodules.items():
    if alias not in sys.modules:
        try:
            import importlib
            mod = importlib.import_module(full_name)
            sys.modules[alias] = mod
        except ImportError:
            pass

# cosyvoice alias 등록 (자기 자신)
if 'cosyvoice' not in sys.modules:
    import vtts.engines._cosyvoice
    sys.modules['cosyvoice'] = vtts.engines._cosyvoice
