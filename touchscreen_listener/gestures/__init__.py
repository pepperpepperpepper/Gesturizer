"""
Gesture detection and classification system.

This module provides functionality for detecting and classifying
touchscreen gestures, including both quick gestures and deliberate drawings.
"""

from .path_classifier import PathClassifier, classify_path, get_path_stats
from .gesture_detector import GestureDetector

__all__ = [
    'PathClassifier',
    'classify_path',
    'get_path_stats',
    'GestureDetector'
]