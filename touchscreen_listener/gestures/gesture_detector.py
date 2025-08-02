"""
Gesture detection and classification system.
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any
from ..config.settings import TouchConfig

class GestureDetector:
    """Detects and classifies touch gestures."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = TouchConfig()
        
        # Calculate pixel values based on screen resolution
        self._calculate_pixel_values()
        
        # Gesture state tracking
        self.tap_sequence = []
        self.last_tap_time = 0
        self.last_tap_fingers = 0
        self.last_tap_positions = []
        
        # Pinch tracking
        self.pinch_data = {
            'initial_distance': 0.0,
            'min_distance': float('inf'),
            'max_distance': 0.0,
            'is_active': False
        }
        
        # Motion tracking for scrub detection
        self.motion_history = {}
    
    def _calculate_pixel_values(self):
        """Calculate pixel values based on screen resolution."""
        screen_diagonal = math.sqrt(self.screen_width**2 + self.screen_height**2)
        
        self.TAP_DISTANCE = int(screen_diagonal * self.config.TAP_DISTANCE_PERCENT / 100)
        self.SWIPE_DISTANCE = int(screen_diagonal * self.config.SWIPE_DISTANCE_PERCENT / 100)
        self.DOUBLE_TAP_MAX_DISTANCE = int(screen_diagonal * self.config.DOUBLE_TAP_MAX_DISTANCE_PERCENT / 100)
        self.DIRECTIONAL_SWIPE_MIN_DISTANCE = int(screen_diagonal * self.config.DIRECTIONAL_SWIPE_MIN_DISTANCE_PERCENT / 100)
        self.TAP_HOLD_MAX_MOVEMENT = int(screen_diagonal * self.config.TAP_HOLD_MAX_MOVEMENT_PERCENT / 100)
        self.SCRUB_MIN_SEGMENT_DISTANCE = int(self.screen_height * self.config.SCRUB_MIN_SEGMENT_PERCENT / 100)
        self.SCRUB_MIN_VERTICAL_TRAVEL = int(self.screen_height * self.config.SCRUB_MIN_VERTICAL_PERCENT / 100)
        self.DRAG_HOLD_MIN_DRAG_DISTANCE = int(screen_diagonal * self.config.DRAG_HOLD_MIN_DRAG_DISTANCE_PERCENT / 100)
        self.DRAG_HOLD_STOP_MOVEMENT = int(screen_diagonal * self.config.DRAG_HOLD_STOP_MOVEMENT_PERCENT / 100)
    
    def classify_gesture(self, fingers: Dict[int, Tuple[int, int, int, int, float]], 
                        is_hold: bool = False, is_drag_hold: bool = False) -> Dict[str, Any]:
        """Classify the gesture based on finger movements and hold state."""
        if not fingers:
            return {'type': 'none'}
        
        finger_count = len(fingers)
        positions = [(x, y) for _, _, x, y, _ in fingers.values()]
        
        # If this is a hold gesture, don't classify as tap
        if is_hold:
            return {
                'type': 'hold',
                'finger_count': finger_count,
                'positions': positions
            }
        
        if is_drag_hold:
            # Calculate drag specifics
            total_distance = 0
            for sx, sy, ex, ey, _ in fingers.values():
                dist = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                total_distance += dist
            avg_distance = total_distance / finger_count if finger_count > 0 else 0
            
            return {
                'type': 'drag_hold',
                'finger_count': finger_count,
                'positions': positions,
                'drag_distance': avg_distance
            }
        
        # Calculate movement
        max_move = 0
        for sx, sy, ex, ey, _ in fingers.values():
            move = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            max_move = max(max_move, move)
        
        # Check for tap (including double/triple) - only if not a hold
        tap_result = self._check_tap_gesture(fingers, max_move)
        if tap_result['is_tap']:
            return {
                'type': 'tap',
                'tap_type': tap_result['tap_type'],
                'finger_count': finger_count,
                'positions': positions
            }
        
        # Check for pinch - only if there's significant movement
        pinch_result = self._check_pinch_gesture(fingers)
        if pinch_result['is_pinch']:
            return {
                'type': 'pinch',
                'pinch_type': pinch_result['pinch_type'],
                'finger_count': finger_count,
                'positions': positions,
                'distance_change': pinch_result['distance_change']
            }
        
        # Check for scrub
        is_scrub = self._check_scrub_gesture(fingers)
        if is_scrub:
            return {
                'type': 'scrub',
                'finger_count': finger_count,
                'positions': positions,
                'segments': is_scrub['segments'],
                'vertical_travel': is_scrub['vertical_travel']
            }
        
        # Check for directional swipe
        directional = self._check_directional_swipe(fingers)
        if directional:
            return directional
        
        # Default to regular swipe
        return self._classify_swipe(fingers)
    
    def _check_tap_gesture(self, fingers: Dict[int, Tuple[int, int, int, int, float]], max_move: float) -> Dict[str, Any]:
        """Check for tap gestures including double and triple taps."""
        if max_move >= self.TAP_DISTANCE:
            return {'is_tap': False}
        
        finger_count = len(fingers)
        positions = [(x, y) for _, _, x, y, _ in fingers.values()]
        current_time = time.time() * 1000
        
        # Add to tap sequence
        self.tap_sequence.append({
            'time': current_time,
            'finger_count': finger_count,
            'positions': positions.copy()
        })
        
        # Clean old taps
        self.tap_sequence = [tap for tap in self.tap_sequence 
                           if current_time - tap['time'] < self.config.TRIPLE_TAP_TIMEOUT]
        
        # Check for triple tap
        if len(self.tap_sequence) >= 3:
            last_three = self.tap_sequence[-3:]
            if (last_three[0]['finger_count'] == last_three[1]['finger_count'] == last_three[2]['finger_count']):
                if self._positions_close(last_three[0]['positions'], last_three[-1]['positions']):
                    self.tap_sequence = []
                    return {
                        'is_tap': True,
                        'tap_type': 'triple',
                        'finger_count': finger_count,
                        'positions': positions
                    }
        
        # Check for double tap
        time_since_last = current_time - self.last_tap_time
        if (time_since_last < self.config.DOUBLE_TAP_TIMEOUT and
            finger_count == self.last_tap_fingers and
            self.last_tap_positions and
            self._positions_close(positions, self.last_tap_positions)):
            
            self.last_tap_time = 0
            self.last_tap_fingers = 0
            self.last_tap_positions = []
            return {
                'is_tap': True,
                'tap_type': 'double',
                'finger_count': finger_count,
                'positions': positions
            }
        
        # Single tap
        self.last_tap_time = current_time
        self.last_tap_fingers = finger_count
        self.last_tap_positions = positions.copy()
        
        return {
            'is_tap': True,
            'tap_type': 'single',
            'finger_count': finger_count,
            'positions': positions
        }
    
    def _check_pinch_gesture(self, fingers: Dict[int, Tuple[int, int, int, int, float]]) -> Dict[str, Any]:
        """Check for pinch gestures."""
        finger_count = len(fingers)
        
        if finger_count < self.config.PINCH_MIN_FINGERS or finger_count > self.config.PINCH_MAX_FINGERS:
            return {'is_pinch': False}
        
        if not self.pinch_data['is_active']:
            return {'is_pinch': False}
        
        # Calculate distance change
        current_distance = self._calculate_distance(fingers)
        if current_distance <= 0:
            return {'is_pinch': False}
        
        # Calculate actual change from initial position
        actual_change = abs(current_distance - self.pinch_data['initial_distance'])
        
        # Require minimum absolute change (not just percentage)
        min_absolute_change = max(
            self.pinch_data['initial_distance'] * (self.config.PINCH_MIN_CHANGE_PERCENT / 100),
            50.0  # Minimum 50 pixels change required
        )
        
        if actual_change < min_absolute_change:
            return {'is_pinch': False}
        
        # Check direction
        pinch_type = self._determine_pinch_direction(fingers, current_distance)
        
        return {
            'is_pinch': True,
            'pinch_type': pinch_type,
            'distance_change': actual_change,
            'initial_distance': self.pinch_data['initial_distance'],
            'final_distance': current_distance
        }
    
    def _check_scrub_gesture(self, fingers: Dict[int, Tuple[int, int, int, int, float]]) -> Optional[Dict[str, Any]]:
        """Check for scrub gestures."""
        if not fingers:
            return None
        
        # Use first finger for scrub detection
        slot = list(fingers.keys())[0]
        
        if slot not in self.motion_history:
            return None
        
        history = self.motion_history[slot]
        
        num_segments = len(history['vertical_segments'])
        if abs(history['segment_dy']) >= self.SCRUB_MIN_SEGMENT_DISTANCE:
            num_segments += 1
        
        if (num_segments >= self.config.SCRUB_MIN_SEGMENTS and
            history['total_vertical_travel'] >= self.SCRUB_MIN_VERTICAL_TRAVEL and
            history['total_vertical_travel'] > history['total_horizontal_travel'] * 1.5):
            
            return {
                'segments': num_segments,
                'vertical_travel': history['total_vertical_travel']
            }
        
        return None
    
    def _check_directional_swipe(self, fingers: Dict[int, Tuple[int, int, int, int, float]]) -> Optional[Dict[str, Any]]:
        """Check for directional swipes."""
        if not fingers:
            return None
        
        # Use first finger
        sx, sy, ex, ey, _ = list(fingers.values())[0]
        dx = ex - sx
        dy = ey - sy
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < self.DIRECTIONAL_SWIPE_MIN_DISTANCE:
            return None
        
        # Calculate swipe angle
        angle = math.degrees(math.atan2(-dy, dx))
        if angle < 0:
            angle += 360
        
        # Check for directional patterns
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Define compass regions
        compass_regions = {
            'N': (center_x, 0),
            'NE': (self.screen_width, 0),
            'E': (self.screen_width, center_y),
            'SE': (self.screen_width, self.screen_height),
            'S': (center_x, self.screen_height),
            'SW': (0, self.screen_height),
            'W': (0, center_y),
            'NW': (0, 0)
        }
        
        # Check if swipe starts near a compass point and heads toward center
        for compass_point, (target_x, target_y) in compass_regions.items():
            start_distance = math.sqrt((sx - target_x)**2 + (sy - target_y)**2)
            start_threshold = min(self.screen_width, self.screen_height) * 0.15
            
            if start_distance < start_threshold:
                center_angle = math.degrees(math.atan2(-(center_y - sy), center_x - sx))
                if center_angle < 0:
                    center_angle += 360
                
                angle_diff = abs(angle - center_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff <= self.config.DIRECTIONAL_SWIPE_ANGLE_TOLERANCE:
                    return {
                        'type': 'directional_swipe',
                        'direction_type': 'TO_CENTER',
                        'compass_point': compass_point,
                        'finger_count': len(fingers),
                        'positions': [(x, y) for _, _, x, y, _ in fingers.values()]
                    }
        
        return None
    
    def _classify_swipe(self, fingers: Dict[int, Tuple[int, int, int, int, float]]) -> Dict[str, Any]:
        """Classify a regular swipe gesture."""
        sx, sy, ex, ey, _ = list(fingers.values())[0]
        dx = ex - sx
        dy = ey - sy
        
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        
        distance = math.sqrt(dx*dx + dy*dy)
        
        return {
            'type': 'swipe',
            'direction': direction,
            'finger_count': len(fingers),
            'positions': [(x, y) for _, _, x, y, _ in fingers.values()],
            'distance': distance
        }
    
    def _positions_close(self, pos1: List[Tuple[int, int]], pos2: List[Tuple[int, int]]) -> bool:
        """Check if two sets of positions are close enough for double/triple tap."""
        if len(pos1) != len(pos2):
            return False
        
        max_distance = 0
        for (x1, y1), (x2, y2) in zip(pos1, pos2):
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            max_distance = max(max_distance, distance)
            if distance > self.DOUBLE_TAP_MAX_DISTANCE:
                return False
        
        return True
    
    def _calculate_distance(self, fingers: Dict[int, Tuple[int, int, int, int, float]]) -> float:
        """Calculate distance between fingers for pinch detection."""
        finger_count = len(fingers)
        
        if finger_count < 2 or finger_count > self.config.PINCH_MAX_FINGERS:
            return 0.0
        
        if finger_count == 2:
            # 2-finger pinch: simple distance between two points
            slots = list(fingers.keys())
            x1, y1 = fingers[slots[0]][2], fingers[slots[0]][3]
            x2, y2 = fingers[slots[1]][2], fingers[slots[1]][3]
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        elif finger_count >= 3:
            # 3-5 finger pinch: use average distance between all pairs
            positions = [(fingers[slot][2], fingers[slot][3]) for slot in fingers.keys()]
            distances = []
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    distances.append(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
            
            return sum(distances) / len(distances) if distances else 0.0
        
        return 0.0
    
    def _determine_pinch_direction(self, fingers: Dict[int, Tuple[int, int, int, int, float]], 
                                   final_distance: float) -> str:
        """Determine the direction of a pinch gesture."""
        finger_count = len(fingers)
        
        if finger_count == 2:
            # Simple comparison for 2-finger pinch
            if final_distance > self.pinch_data['initial_distance']:
                return 'ZOOM_IN'
            else:
                return 'ZOOM_OUT'
        
        else:
            # For 3+ fingers, use center-based approach
            slots = list(fingers.keys())
            positions = [(fingers[slot][2], fingers[slot][3]) for slot in slots]
            start_positions = [(fingers[slot][0], fingers[slot][1]) for slot in slots]
            
            # Calculate average distance from center
            center_x = sum(x for x, y in positions) / finger_count
            center_y = sum(y for x, y in positions) / finger_count
            start_center_x = sum(x for x, y in start_positions) / finger_count
            start_center_y = sum(y for x, y in start_positions) / finger_count
            
            start_avg_distance = 0
            end_avg_distance = 0
            
            for i, (sx, sy) in enumerate(start_positions):
                start_avg_distance += math.sqrt((sx - start_center_x)**2 + (sy - start_center_y)**2)
                end_avg_distance += math.sqrt((positions[i][0] - center_x)**2 + (positions[i][1] - center_y)**2)
            
            start_avg_distance /= finger_count
            end_avg_distance /= finger_count
            
            if end_avg_distance > start_avg_distance:
                return 'ZOOM_OUT'
            else:
                return 'ZOOM_IN'
    
    def update_pinch_tracking(self, fingers: Dict[int, Tuple[int, int, int, int, float]]):
        """Update pinch tracking state."""
        finger_count = len(fingers)
        
        if finger_count >= self.config.PINCH_MIN_FINGERS and finger_count <= self.config.PINCH_MAX_FINGERS:
            current_distance = self._calculate_distance(fingers)
            if current_distance > 0:
                if not self.pinch_data['is_active']:
                    # Only activate pinch tracking after meaningful finger movement
                    has_moved = any(
                        abs(ex - sx) > 10 or abs(ey - sy) > 10
                        for sx, sy, ex, ey, _ in fingers.values()
                    )
                    if has_moved:
                        self.pinch_data['initial_distance'] = current_distance
                        self.pinch_data['min_distance'] = current_distance
                        self.pinch_data['max_distance'] = current_distance
                        self.pinch_data['is_active'] = True
                else:
                    self.pinch_data['min_distance'] = min(self.pinch_data['min_distance'], current_distance)
                    self.pinch_data['max_distance'] = max(self.pinch_data['max_distance'], current_distance)
            else:
                self.pinch_data['is_active'] = False
        else:
            self.pinch_data['is_active'] = False
    
    def track_motion(self, slot: int, x: int, y: int):
        """Track finger motion for scrub detection."""
        if slot not in self.motion_history:
            self.motion_history[slot] = {
                'vertical_segments': [],
                'segment_dy': 0,
                'total_vertical_travel': 0,
                'total_horizontal_travel': 0,
                'last_x': x,
                'last_y': y
            }
        
        history = self.motion_history[slot]
        
        # Add new position
        dx = x - history['last_x']
        dy = y - history['last_y']
        history['total_vertical_travel'] += abs(dy)
        history['total_horizontal_travel'] += abs(dx)
        history['segment_dy'] += dy
        
        current_sign = 1 if history['segment_dy'] > 0 else -1 if history['segment_dy'] < 0 else 0
        new_sign = 1 if dy > 0 else -1 if dy < 0 else 0
        
        if new_sign != 0 and current_sign != 0 and new_sign != current_sign:
            if abs(history['segment_dy']) >= self.SCRUB_MIN_SEGMENT_DISTANCE:
                direction = 'down' if history['segment_dy'] > 0 else 'up'
                history['vertical_segments'].append(direction)
                history['segment_dy'] = dy
        
        history['last_x'] = x
        history['last_y'] = y
    
    def cleanup_motion_history(self):
        """Clean up motion history for all slots."""
        self.motion_history.clear()