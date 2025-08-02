"""
Configuration settings for the touchscreen listener.
"""

class TouchConfig:
    """Configuration constants for touch gesture recognition."""
    
    # Timing configurations (in milliseconds)
    TRIPLE_TAP_TIMEOUT = 1500
    DOUBLE_TAP_TIMEOUT = 800
    TAP_HOLD_TIMEOUT = 500
    TAP_HOLD_MIN_DURATION = 500
    DRAG_HOLD_TIMEOUT = 500
    
    # Distance configurations (as percentages of screen diagonal/height)
    TAP_DISTANCE_PERCENT = 3.0
    SWIPE_DISTANCE_PERCENT = 15.0
    DOUBLE_TAP_MAX_DISTANCE_PERCENT = 8.0
    DIRECTIONAL_SWIPE_MIN_DISTANCE_PERCENT = 10.0
    TAP_HOLD_MAX_MOVEMENT_PERCENT = 0.5
    DRAG_HOLD_MIN_DRAG_DISTANCE_PERCENT = 5.0
    DRAG_HOLD_STOP_MOVEMENT_PERCENT = 2.0  # Temporarily increased for debugging
    
    # Scrub detection
    SCRUB_MIN_SEGMENTS = 4
    SCRUB_MIN_SEGMENT_PERCENT = 3.0
    SCRUB_MIN_VERTICAL_PERCENT = 20.0
    
    # Pinch detection
    PINCH_MIN_FINGERS = 2
    PINCH_MAX_FINGERS = 5
    PINCH_MIN_CHANGE_PERCENT = 30.0
    PINCH_DIRECTION_ANGLE_TOLERANCE = 45
    
    # Directional swipe
    DIRECTIONAL_SWIPE_ANGLE_TOLERANCE = 30
    
    # Compass points for directional swipes
    COMPASS_POINTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    COMPASS_ANGLES = {
        'N': 90, 'NE': 45, 'E': 0, 'SE': 315, 
        'S': 270, 'SW': 225, 'W': 180, 'NW': 135
    }