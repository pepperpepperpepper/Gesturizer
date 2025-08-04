"""
Utilities package for gesture recognition and processing.

This package provides shared utilities to eliminate code duplication
across different classifiers and gesture processing components.
"""

from .gesture_utils import (
    Point,
    GeometryUtils,
    VelocityCalculator,
    PathUtils,
    DataValidator
)

__all__ = [
    'Point',
    'GeometryUtils', 
    'VelocityCalculator',
    'PathUtils',
    'DataValidator'
]