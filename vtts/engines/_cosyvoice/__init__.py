"""CosyVoice TTS Engine - Embedded Implementation

자동으로 matcha 및 cosyvoice 모듈 alias를 등록합니다.
"""
import sys
import importlib

def _setup_module_aliases():
    """matcha 및 cosyvoice 모듈 alias를 sys.modules에 등록"""
    # 현재 패키지 경로
    pkg_path = 'vtts.engines._cosyvoice'
    
    # matcha 최상위 모듈 등록
    if 'matcha' not in sys.modules:
        try:
            matcha_mod = importlib.import_module(f'{pkg_path}.matcha')
            sys.modules['matcha'] = matcha_mod
        except ImportError:
            pass
    
    # matcha 하위 모듈 등록
    matcha_submodules = [
        'models',
        'models.components',
        'models.components.flow_matching',
        'models.components.decoder',
        'models.components.transformer',
        'hifigan',
        'hifigan.models',
        'hifigan.xutils',
    ]
    
    for submod in matcha_submodules:
        alias = f'matcha.{submod}'
        if alias not in sys.modules:
            try:
                mod = importlib.import_module(f'{pkg_path}.matcha.{submod}')
                sys.modules[alias] = mod
            except ImportError:
                pass
    
    # cosyvoice alias는 나중에 load_model에서 등록

# 모듈 로드 시 자동 실행
_setup_module_aliases()
