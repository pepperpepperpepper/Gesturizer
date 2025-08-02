"""
Logging utilities for touch events and gestures.
"""

import datetime
import logging
import time
from typing import Dict, Any, List, Tuple

class TouchLogger:
    """Handles logging of touch events and gestures."""
    
    def __init__(self, debug_file: str = 'double_tap_debug.log'):
        self.debug_file = None
        try:
            self.debug_file = open(debug_file, 'w')
            self.debug_file.write(f"Debug logging started at {datetime.datetime.now()}\n")
            self.debug_file.flush()
        except Exception as e:
            print(f"Warning: Could not open debug file: {e}")
    
    def log_gesture(self, gesture: Dict[str, Any]):
        """Log a detected gesture."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        gesture_type = gesture.get('type', 'unknown')
        finger_count = gesture.get('finger_count', 0)
        positions = gesture.get('positions', [])
        
        if gesture_type == 'tap':
            tap_type = gesture.get('tap_type', 'single')
            if tap_type == 'triple':
                print(f"[{timestamp}] 👆👆👆 TRIPLE TAP: {finger_count} finger(s)")
            elif tap_type == 'double':
                print(f"[{timestamp}] 👆👆 DOUBLE TAP: {finger_count} finger(s)")
            else:
                print(f"[{timestamp}] 👆 TAP: {finger_count} finger(s)")
        
        elif gesture_type == 'hold':
            print(f"[{timestamp}] 🤚 TOUCH AND HOLD: {finger_count} finger(s)")
        
        elif gesture_type == 'drag_hold':
            drag_distance = gesture.get('drag_distance', 0)
            print(f"[{timestamp}] 🖐️ DRAG AND HOLD: {finger_count} finger(s) [Drag distance: {int(drag_distance)}px]")
        
        elif gesture_type == 'swipe':
            direction = gesture.get('direction', 'unknown')
            distance = gesture.get('distance', 0)
            print(f"[{timestamp}] 👋 SWIPE: {finger_count} finger(s) {direction} [{int(distance)}px]")
        
        elif gesture_type == 'directional_swipe':
            direction_type = gesture.get('direction_type', 'unknown')
            compass_point = gesture.get('compass_point', 'unknown')
            print(f"[{timestamp}] 🧭 DIRECTIONAL SWIPE: {finger_count} finger(s) {direction_type} {compass_point}")
        
        elif gesture_type == 'pinch':
            pinch_type = gesture.get('pinch_type', 'unknown')
            distance_change = gesture.get('distance_change', 0)
            initial_distance = gesture.get('initial_distance', 0)
            final_distance = gesture.get('final_distance', 0)
            
            print(f"[{timestamp}] 🔍 PINCH {pinch_type}: {finger_count} finger(s)")
            print(f"   Initial distance: {initial_distance:.1f}px, Final distance: {final_distance:.1f}px")
            change_percent = (distance_change / initial_distance) * 100 if initial_distance > 0 else 0
            print(f"   Distance change: {change_percent:.1f}%")
        
        elif gesture_type == 'scrub':
            segments = gesture.get('segments', 0)
            vertical_travel = gesture.get('vertical_travel', 0)
            print(f"[{timestamp}] 🔄 SCRUB GESTURE: {finger_count} finger(s)")
            print(f"   Vertical segments: {segments}")
            print(f"   Total vertical travel: {int(vertical_travel)}px")
        
        # Print finger positions
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(x)}, {int(y)})")
        
        # Debug file logging
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] {gesture}\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def log_hold_start(self, finger_count: int, positions: List[Tuple[int, int]], start_time: float):
        """Log tap and hold start."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        confirmation_time = time.time()
        time_to_confirm = (confirmation_time - start_time) * 1000
        
        print(f"[{timestamp}] 🤚 TOUCH AND HOLD CONFIRMED: {finger_count} finger(s)")
        print(f"   Time to confirm: {time_to_confirm:.0f}ms")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
    
    def log_hold_end(self, finger_count: int, positions: List[Tuple[int, int]], start_time: float):
        """Log tap and hold end."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hold_duration = time.time() - start_time
        start_timestamp = datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S.%f")[:-3]
        
        print(f"[{timestamp}] ✋ TOUCH AND HOLD END: {finger_count} finger(s)")
        print(f"   Started at: {start_timestamp}")
        print(f"   Total duration: {hold_duration:.1f}s")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
    
    def log_drag_hold_start(self, finger_count: int, positions: List[Tuple[int, int]], 
                           start_positions: List[Tuple[int, int]], start_time: float):
        """Log drag and hold start."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        confirmation_time = time.time()
        time_to_confirm = (confirmation_time - start_time) * 1000
        
        print(f"[{timestamp}] 🖐️ DRAG AND HOLD CONFIRMED: {finger_count} finger(s)")
        print(f"   Time to confirm hold after drag: {time_to_confirm:.0f}ms")
        
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1} hold position: ({int(x)}, {int(y)})")
        
        # Calculate drag distance for each finger
        for i, (sx, sy) in enumerate(start_positions):
            ex, ey = positions[i]
            dist = ((ex - sx)**2 + (ey - sy)**2)**0.5
            print(f"   Finger {i+1} drag distance: {int(dist)}px")
    
    def log_drag_hold_end(self, finger_count: int, positions: List[Tuple[int, int]], 
                        drag_start_time: float, stop_start_time: float):
        """Log drag and hold end."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        end_time = time.time()
        total_duration = end_time - drag_start_time
        hold_duration = end_time - stop_start_time
        
        print(f"[{timestamp}] ✊ DRAG AND HOLD END: {finger_count} finger(s)")
        print(f"   Total duration: {total_duration:.1f}s (drag + hold)")
        print(f"   Hold duration: {hold_duration:.1f}s")
        
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1} end position: ({int(x)}, {int(y)})")
    
    def close(self):
        """Close the debug file."""
        if self.debug_file:
            self.debug_file.close()