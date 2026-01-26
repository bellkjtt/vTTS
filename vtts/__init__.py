"""vTTS - Universal TTS Serving System

vLLM for Text-to-Speech
"""

__version__ = "0.2.0"

import os
import sys
import warnings

# ============================================================
# CUDA 환경 자동 설정
# ============================================================
def _setup_cuda_environment():
    """CUDA 환경을 자동으로 설정합니다. (cuDNN 버전 충돌 해결)"""
    try:
        import torch
        
        # PyTorch가 CUDA를 지원하는 경우에만 설정
        if not torch.cuda.is_available():
            return
        
        # PyTorch 번들 cuDNN 경로를 LD_LIBRARY_PATH 앞에 추가
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        
        if os.path.exists(torch_lib):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            
            # 이미 설정되어 있으면 건너뛰기
            if torch_lib not in current_ld_path:
                # NVIDIA 패키지 라이브러리 경로도 추가
                nvidia_paths = []
                site_packages = os.path.dirname(os.path.dirname(torch.__file__))
                nvidia_dir = os.path.join(site_packages, "nvidia")
                
                if os.path.exists(nvidia_dir):
                    for pkg in os.listdir(nvidia_dir):
                        lib_path = os.path.join(nvidia_dir, pkg, "lib")
                        if os.path.exists(lib_path):
                            nvidia_paths.append(lib_path)
                
                # 새 LD_LIBRARY_PATH 구성
                new_paths = [torch_lib] + nvidia_paths
                new_ld_path = ":".join(new_paths)
                
                if current_ld_path:
                    # 시스템 cuDNN 경로 제거 (충돌 방지)
                    filtered_paths = [
                        p for p in current_ld_path.split(":")
                        if "cudnn" not in p.lower() and p not in new_paths
                    ]
                    new_ld_path = new_ld_path + ":" + ":".join(filtered_paths)
                
                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                
    except ImportError:
        pass
    except Exception:
        pass  # 조용히 실패


def _check_onnxruntime_cuda():
    """ONNX Runtime CUDA 지원을 확인하고 필요시 설치합니다."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        # CUDAExecutionProvider가 없으면 CUDA 12 버전 설치 안내
        if "CUDAExecutionProvider" not in providers:
            # torch가 CUDA를 지원하는지 확인
            try:
                import torch
                if torch.cuda.is_available():
                    warnings.warn(
                        "vTTS: ONNX Runtime에 CUDA 지원이 없습니다. "
                        "Supertonic CUDA 사용을 위해 다음 명령 실행:\n"
                        "pip install onnxruntime-gpu --extra-index-url "
                        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
                        UserWarning,
                        stacklevel=2
                    )
            except ImportError:
                pass
    except ImportError:
        pass
    except Exception:
        pass


# CUDA 환경 자동 설정 (import 시 실행)
_setup_cuda_environment()

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
                f"numpy 2.x는 일부 엔진과 호환성 문제가 있을 수 있습니다.",
                UserWarning,
                stacklevel=2
            )
    except ImportError:
        pass
    
    # ONNX Runtime CUDA 확인
    _check_onnxruntime_cuda()

# 환경 확인 실행
_check_environment()

# ============================================================
# Public API
# ============================================================
from vtts.client import VTTSClient
from vtts.engines.registry import EngineRegistry

__all__ = ["VTTSClient", "EngineRegistry", "__version__"]
