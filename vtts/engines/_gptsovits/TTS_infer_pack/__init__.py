import os
import sys

# 내장된 _gptsovits 패키지의 루트 경로를 먼저 설정
_gptsovits_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gptsovits_root not in sys.path:
    sys.path.insert(0, _gptsovits_root)

from . import TTS, text_segmentation_method
