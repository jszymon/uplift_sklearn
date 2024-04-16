"""
The :mod:`usklearn.utils` module includes various utilities.
"""

from .validation import check_trt
from .multi_array import MultiArray
from .array_utils import safe_hstack

__all__ = ["check_trt", "MultiArray"]
