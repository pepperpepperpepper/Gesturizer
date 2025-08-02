"""
Touchscreen Listener Package
A modular touchscreen event listener for gesture recognition.
"""

from .core.listener import TouchListener
from .gestures.gesture_detector import GestureDetector
from .device.device_manager import DeviceManager

__version__ = "2.0.0"
__all__ = ["TouchListener", "GestureDetector", "DeviceManager"]