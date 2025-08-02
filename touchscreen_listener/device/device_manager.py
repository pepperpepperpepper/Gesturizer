"""
Device management for touchscreen discovery and initialization.
"""

import evdev
from evdev import InputDevice, ecodes
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages touchscreen device discovery and initialization."""
    
    def __init__(self):
        self.device = None
        self.screen_width = 1920  # Default
        self.screen_height = 1080   # Default
        
    def find_device(self):
        """Find and configure the touchscreen device."""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        
        for device in devices:
            caps = device.capabilities()
            if ecodes.EV_ABS in caps:
                # Check for multitouch capabilities
                abs_caps = caps.get(ecodes.EV_ABS, [])
                abs_codes = [code for code, _ in abs_caps]
                
                # Look for multitouch slots
                if ecodes.ABS_MT_SLOT in abs_codes:
                    # Get screen resolution from device capabilities
                    abs_info = {code: info for code, info in abs_caps}
                    
                    if ecodes.ABS_MT_POSITION_X in abs_info:
                        self.screen_width = abs_info[ecodes.ABS_MT_POSITION_X].max + 1
                    if ecodes.ABS_MT_POSITION_Y in abs_info:
                        self.screen_height = abs_info[ecodes.ABS_MT_POSITION_Y].max + 1
                        
                    self.device = device
                    logger.info(f"Found touchscreen: {device.name}")
                    logger.info(f"Screen resolution: {self.screen_width}x{self.screen_height}")
                    return device
        
        logger.error("No touchscreen device found")
        return None
    
    def get_device_info(self):
        """Get device and screen information."""
        return {
            'device': self.device,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'center_x': self.screen_width // 2,
            'center_y': self.screen_height // 2
        }