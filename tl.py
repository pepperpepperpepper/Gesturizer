#!/usr/bin/env python3
"""
Touchscreen Listener (tl.py)
A relaxed touchscreen event listener that actually works.
"""

import evdev
from evdev import InputDevice, ecodes
import time
import threading
import math

class TouchListener:
    def __init__(self):
        self.device = None
        self.running = False
        self.fingers = {}
        self.last_tap = None
        self.last_tap_time = 0
        self.last_tap_fingers = 0
        self.last_tap_positions = []
        
        # Triple tap tracking
        self.tap_sequence = []  # Store sequence of taps for triple tap detection
        self.TRIPLE_TAP_TIMEOUT = 1500  # 1500ms for triple tap sequence
        
        # Screen resolution (will be detected)
        self.screen_width = 1920  # Default, will be updated
        self.screen_height = 1080  # Default, will be updated
        
        # Percentage-based parameters
        self.TAP_DISTANCE_PERCENT = 3.0  # 3% of screen diagonal
        self.SWIPE_DISTANCE_PERCENT = 15.0  # 15% of screen diagonal
        self.DOUBLE_TAP_MAX_DISTANCE_PERCENT = 8.0  # 8% of screen diagonal
        self.DOUBLE_TAP_TIMEOUT = 800  # 800ms for comfortable double tapping
        
        # Directional swipe detection
        self.DIRECTIONAL_SWIPE_ANGLE_TOLERANCE = 30  # degrees tolerance for directional swipes
        self.DIRECTIONAL_SWIPE_MIN_DISTANCE_PERCENT = 10.0  # minimum distance as % of screen diagonal
        self.compass_points = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.compass_angles = {
            'N': 90, 'NE': 45, 'E': 0, 'SE': 315, 
            'S': 270, 'SW': 225, 'W': 180, 'NW': 135
        }
        
        # Tap and hold detection
        self.TAP_HOLD_TIMEOUT = 1000  # 1000ms (1 second) before recognizing as hold
        self.TAP_HOLD_MAX_MOVEMENT_PERCENT = 1.5  # Reduced to 1.5% for better swipe/hold distinction
        self.TAP_HOLD_MIN_DURATION = 500  # Minimum 500ms before considering as hold (prevent false positives)
        self.active_holds = {}  # Track active holds by slot
        self.gesture_hold_state = {
            'is_hold': False,
            'start_time': 0,
            'notified': False,
            'start_positions': []
        }  # Track hold state per gesture (all fingers together)
        
        # Real-time scrub detection (new approach)
        self.motion_history = {}  # Track motion history for each slot
        self.SCRUB_MIN_DIRECTION_CHANGES = 2  # Minimum direction changes for scrub
        self.SCRUB_MIN_TOTAL_DISTANCE_PERCENT = 20.0  # Minimum total distance for scrub
        
        # Debug logging
        try:
            self.debug_file = open('double_tap_debug.log', 'w')
            import datetime
            self.debug_file.write(f"Debug logging started at {datetime.datetime.now()}\n")
            self.debug_file.flush()
        except Exception as e:
            print(f"Warning: Could not open debug file: {e}")
            self.debug_file = None
    
    def find_device(self):
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
                    return device
        return None
    
    def start(self):
        self.device = self.find_device()
        if not self.device:
            print("❌ No touchscreen found")
            return False
        
        # Calculate pixel values based on screen resolution
        screen_diagonal = math.sqrt(self.screen_width**2 + self.screen_height**2)
        self.TAP_DISTANCE = int(screen_diagonal * self.TAP_DISTANCE_PERCENT / 100)
        self.SWIPE_DISTANCE = int(screen_diagonal * self.SWIPE_DISTANCE_PERCENT / 100)
        self.DOUBLE_TAP_MAX_DISTANCE = int(screen_diagonal * self.DOUBLE_TAP_MAX_DISTANCE_PERCENT / 100)
        self.DIRECTIONAL_SWIPE_MIN_DISTANCE = int(screen_diagonal * self.DIRECTIONAL_SWIPE_MIN_DISTANCE_PERCENT / 100)
        self.TAP_HOLD_MAX_MOVEMENT = int(screen_diagonal * self.TAP_HOLD_MAX_MOVEMENT_PERCENT / 100)
        self.SCRUB_MIN_TOTAL_DISTANCE = int(screen_diagonal * self.SCRUB_MIN_TOTAL_DISTANCE_PERCENT / 100)
        
        # Remove old scrub variables (we're using motion-based approach now)
        self.scrub_swipes = []  # Keep for compatibility but will be removed
        self.SCRUB_MIN_SWIPES = 2  # Not used in new approach
        self.SCRUB_TIMEOUT = 4000  # Not used in new approach
        self.SCRUB_MAX_SWIPE_INTERVAL = 1500  # Not used in new approach
        self.SCRUB_MIN_SWIPE_DISTANCE = 0  # Not used in new approach
        self.last_scrub_time = 0  # Not used in new approach
        
        # Calculate screen center and compass regions
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        # Define compass regions (corners and edges)
        self.compass_regions = {
            'N': (self.center_x, 0),
            'NE': (self.screen_width, 0),
            'E': (self.screen_width, self.center_y),
            'SE': (self.screen_width, self.screen_height),
            'S': (self.center_x, self.screen_height),
            'SW': (0, self.screen_height),
            'W': (0, self.center_y),
            'NW': (0, 0)
        }
            
        self.running = True
        print(f"✅ Found: {self.device.name}")
        print(f"📺 Screen: {self.screen_width}x{self.screen_height}")
        print(f"📏 Tap threshold: {self.TAP_DISTANCE}px ({self.TAP_DISTANCE_PERCENT}%)")
        print(f"📏 Swipe threshold: {self.SWIPE_DISTANCE}px ({self.SWIPE_DISTANCE_PERCENT}%)")
        print(f"📏 Double tap distance: {self.DOUBLE_TAP_MAX_DISTANCE}px ({self.DOUBLE_TAP_MAX_DISTANCE_PERCENT}%)")
        print(f"📏 Directional swipe min distance: {self.DIRECTIONAL_SWIPE_MIN_DISTANCE}px ({self.DIRECTIONAL_SWIPE_MIN_DISTANCE_PERCENT}%)")
        print(f"🎯 Screen center: ({self.center_x}, {self.center_y})")
        print("🎯 Ready! Try taps, swipes, and directional swipes from corners/edges to center!")
        
        self.thread = threading.Thread(target=self._event_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        if hasattr(self, 'debug_file'):
            self.debug_file.close()
    
    def _event_loop(self):
        try:
            current_slot = 0
            slot_data = {}
            
            for event in self.device.read_loop():
                if not self.running:
                    break
                    
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_MT_SLOT:
                        current_slot = event.value
                    elif event.code == ecodes.ABS_MT_TRACKING_ID:
                        slot = current_slot
                        if event.value == -1:
                            # Finger lifted
                            if slot in self.fingers:
                                # Remove from slot-level hold tracking
                                if slot in self.active_holds:
                                    del self.active_holds[slot]
                                
# Store all finger data before removing this finger
                                all_fingers_data = dict(self.fingers)  # Copy all finger data
                                finger_count_before_release = len(self.fingers)
                                
                                # Remove the finger from tracking
                                del self.fingers[slot]
                                
                                # Check if this was the last finger in the gesture
                                if len(self.fingers) == 0:  # This was the last finger being lifted
                                    if self.gesture_hold_state['is_hold']:
                                        # This was a hold gesture
                                        self._handle_hold_end()
                                    else:
                                        # This was a normal gesture - restore all fingers for processing
                                        self.fingers = all_fingers_data  # Restore all finger data
                                        self._process_gesture(slot)
                                        # Clear all fingers after processing
                                        self.fingers.clear()
                                    
                                    # Reset gesture hold state
                                    self.gesture_hold_state = {
                                        'is_hold': False,
                                        'start_time': 0,
                                        'notified': False,
                                        'start_positions': []
                                    }
                        else:
                            # Finger placed
                            if slot not in slot_data:
                                slot_data[slot] = {'x': 0, 'y': 0}
                            self.fingers[slot] = (0, 0, 0, 0, time.time())
                            
                            # Initialize gesture hold state if this is the first finger
                            if len(self.fingers) == 1:
                                self.gesture_hold_state = {
                                    'is_hold': False,
                                    'start_time': time.time(),
                                    'notified': False,
                                    'start_positions': []
                                }
                                # First finger touched, initializing hold state
                            else:
                                print(f"🔍 Additional finger touched, total fingers: {len(self.fingers)}")
                            
                            # Start tracking for potential hold (per slot)
                            self.active_holds[slot] = {
                                'start_time': time.time(),
                                'start_x': 0,
                                'start_y': 0,
                                'notified': False
                            }                
                elif event.code == ecodes.ABS_MT_POSITION_X:
                    slot = current_slot
                    if slot in slot_data:
                        slot_data[slot]['x'] = event.value
                        if slot in self.fingers:
                            sx, sy, _, old_y, st = self.fingers[slot]
                            if sx == 0:
                                self.fingers[slot] = (event.value, old_y, event.value, old_y, st)
                            else:
                                self.fingers[slot] = (sx, sy, event.value, old_y, st)
                                # Track motion for scrub detection
                                self._track_motion(slot, event.value, slot_data[slot]['y'])
                        # Update hold tracking
                        if slot in self.active_holds:
                            if self.active_holds[slot]['start_x'] == 0:
                                self.active_holds[slot]['start_x'] = event.value
                            else:
                                # Check if movement is too much for a hold
                                movement = abs(event.value - self.active_holds[slot]['start_x'])
                                if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                    # Too much movement, remove from hold tracking
                                    del self.active_holds[slot]
                
                elif event.code == ecodes.ABS_MT_POSITION_Y:
                    slot = current_slot
                    if slot in slot_data:
                        slot_data[slot]['y'] = event.value
                        if slot in self.fingers:
                            sx, sy, old_x, _, st = self.fingers[slot]
                            if sy == 0:
                                self.fingers[slot] = (old_x, event.value, old_x, event.value, st)
                            else:
                                self.fingers[slot] = (sx, sy, old_x, event.value, st)
                                # Track motion for scrub detection
                                self._track_motion(slot, slot_data[slot]['x'], event.value)
                        # Update hold tracking
                        if slot in self.active_holds:
                            if self.active_holds[slot]['start_y'] == 0:
                                self.active_holds[slot]['start_y'] = event.value
                            else:
                                # Check if movement is too much for a hold
                                movement = abs(event.value - self.active_holds[slot]['start_y'])
                                if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                    # Too much movement, remove from hold tracking
                                    del self.active_holds[slot]
                
                # Check for tap and hold (check periodically)
                current_time = time.time()
                if self.gesture_hold_state['start_time'] > 0 and not self.gesture_hold_state['notified']:
                    hold_duration = current_time - self.gesture_hold_state['start_time']
                    hold_duration_ms = hold_duration * 1000
                    
                    if hold_duration_ms >= self.TAP_HOLD_TIMEOUT:  # Use milliseconds directly
                        # Check if no fingers have moved too much
                        all_fingers_stable = True
                        for slot in self.fingers:
                            if slot in self.fingers:
                                sx, sy, ex, ey, st = self.fingers[slot]
                                movement = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                                if movement > self.TAP_HOLD_MAX_MOVEMENT:
                                    all_fingers_stable = False
                                    break
                        
                        if all_fingers_stable:
                            # This is a hold gesture
                            self.gesture_hold_state['is_hold'] = True
                            self.gesture_hold_state['notified'] = True
                            self._handle_hold_start()
                        else:
                            # If moved too much, this is not a hold
                            self.gesture_hold_state['is_hold'] = False
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _process_gesture(self, lifted_slot):
        if not self.fingers:
            return
        
        finger_count = len(self.fingers)
        positions = [(x, y) for _, _, x, y, _ in self.fingers.values()]
        
        # Calculate movement
        max_move = 0
        for sx, sy, ex, ey, _ in self.fingers.values():
            move = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            max_move = max(max_move, move)
        
        # Check for scrub motion within this touch
        is_scrub = self._check_scrub_motion(lifted_slot)
        
        # Simple classification
        if max_move < self.TAP_DISTANCE:
            self._handle_tap(finger_count, positions)
        elif is_scrub:
            self._handle_scrub_gesture(finger_count, positions)
        else:
            self._handle_swipe(finger_count, positions)
        
        # Cleanup motion history for this slot
        if lifted_slot in self.motion_history:
            del self.motion_history[lifted_slot]
        self.fingers.clear()
    
    def _handle_tap(self, finger_count, positions):
        """Handle tap with double and triple tap detection."""
        current_time = time.time() * 1000
        time_since_last = current_time - self.last_tap_time
        
        # Get timestamp for screen and file logging
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Debug logging with timestamp
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] TAP: {finger_count} fingers at {positions}, time_since_last: {time_since_last:.1f}ms\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
        
        # Add tap to sequence for triple tap detection
        self.tap_sequence.append({
            'time': current_time,
            'finger_count': finger_count,
            'positions': positions.copy()
        })
        
        # Clean old taps from sequence (older than TRIPLE_TAP_TIMEOUT)
        self.tap_sequence = [tap for tap in self.tap_sequence 
                           if current_time - tap['time'] < self.TRIPLE_TAP_TIMEOUT]
        
        # Check for triple tap
        is_triple = False
        if len(self.tap_sequence) >= 3:
            # Check if last 3 taps have same finger count and are close enough
            last_three = self.tap_sequence[-3:]
            if (last_three[0]['finger_count'] == last_three[1]['finger_count'] == last_three[2]['finger_count']):
                # Check positions are close enough
                positions_close = True
                max_distance = 0
                
                for i in range(1, 3):
                    if len(last_three[i]['positions']) == len(last_three[0]['positions']):
                        for (x1, y1), (x2, y2) in zip(last_three[i]['positions'], last_three[0]['positions']):
                            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                            max_distance = max(max_distance, distance)
                            if distance > self.DOUBLE_TAP_MAX_DISTANCE:
                                positions_close = False
                                break
                    else:
                        positions_close = False
                    if not positions_close:
                        break
                
                if positions_close:
                    is_triple = True
                    print(f"[{timestamp}] 🔍 TRIPLE TAP CHECK: fingers_ok=True, max_distance={max_distance:.1f}px (<{self.DOUBLE_TAP_MAX_DISTANCE}px)")
                    if self.debug_file:
                        try:
                            debug_msg = f"[{timestamp}] TRIPLE TAP CHECK: fingers_ok=True, positions_ok=True, max_distance={max_distance:.1f}px\n"
                            self.debug_file.write(debug_msg)
                            self.debug_file.flush()
                        except:
                            pass
        
        if is_triple:
            print(f"[{timestamp}] 👆👆👆 TRIPLE TAP: {finger_count} finger(s)")
            if self.debug_file:
                try:
                    self.debug_file.write(f"[{timestamp}] ✅ TRIPLE TAP SUCCESS: {finger_count} fingers\n\n")
                    self.debug_file.flush()
                except:
                    pass
            # Clear sequence after triple tap
            self.tap_sequence = []
            self.last_tap_time = 0
            self.last_tap_fingers = 0
            self.last_tap_positions = []
        else:
            # Check for double tap (only if not part of triple tap sequence)
            is_double = False
            if (time_since_last < self.DOUBLE_TAP_TIMEOUT and
                finger_count == self.last_tap_fingers and
                self.last_tap_positions):
                
                # Check if positions are close enough
                positions_close = True
                max_distance = 0
                if len(positions) == len(self.last_tap_positions):
                    for (x1, y1), (x2, y2) in zip(positions, self.last_tap_positions):
                        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        max_distance = max(max_distance, distance)
                        if distance > self.DOUBLE_TAP_MAX_DISTANCE:
                            positions_close = False
                            break
                else:
                    positions_close = False
                
                # Screen debug output for double tap check
                print(f"[{timestamp}] 🔍 DOUBLE TAP CHECK: time={time_since_last:.1f}ms (<{self.DOUBLE_TAP_TIMEOUT}ms), distance={max_distance:.1f}px (<{self.DOUBLE_TAP_MAX_DISTANCE}px)")
                
                if self.debug_file:
                    try:
                        debug_msg = f"[{timestamp}] DOUBLE TAP CHECK: timeout_ok={time_since_last < self.DOUBLE_TAP_TIMEOUT} ({time_since_last:.1f}ms < {self.DOUBLE_TAP_TIMEOUT}ms), fingers_ok={finger_count == self.last_tap_fingers}, positions_ok={positions_close} ({max_distance:.1f}px < {self.DOUBLE_TAP_MAX_DISTANCE}px)\n"
                        self.debug_file.write(debug_msg)
                        self.debug_file.flush()
                    except:
                        pass
                
                is_double = positions_close
            
            if is_double:
                print(f"[{timestamp}] 👆👆 DOUBLE TAP: {finger_count} finger(s)")
                if self.debug_file:
                    try:
                        self.debug_file.write(f"[{timestamp}] ✅ DOUBLE TAP SUCCESS: {finger_count} fingers\n\n")
                        self.debug_file.flush()
                    except:
                        pass
                self.last_tap_time = 0  # Reset
                self.last_tap_fingers = 0
                self.last_tap_positions = []
            else:
                print(f"[{timestamp}] 👆 TAP: {finger_count} finger(s)")
                if self.debug_file:
                    try:
                        self.debug_file.write(f"[{timestamp}] ❌ SINGLE TAP: storing for future double/triple tap check\n\n")
                        self.debug_file.flush()
                    except:
                        pass
                self.last_tap_time = current_time
                self.last_tap_fingers = finger_count
                self.last_tap_positions = positions.copy()
        
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(x)}, {int(y)})")
    
    def _handle_swipe(self, finger_count, positions):
        """Handle swipe detection with directional swipe support."""
        # Get the first finger's movement
        sx, sy, ex, ey, _ = list(self.fingers.values())[0]
        dx = ex - sx
        dy = ey - sy
        distance = math.sqrt(dx*dx + dy*dy)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Determine swipe direction
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        
        # Check for directional swipes (takes precedence over normal swipes)
        directional_swipe = self._check_directional_swipe(sx, sy, ex, ey, distance)
        if directional_swipe:
            direction_type, compass_point = directional_swipe
            print(f"[{timestamp}] 🧭 DIRECTIONAL SWIPE: {finger_count} finger(s) {direction_type} {compass_point} [{int(distance)}px]")
            return
        
        # If not a directional swipe, do normal swipe detection
        print(f"[{timestamp}] 👋 SWIPE: {finger_count} finger(s) {direction} [{int(distance)}px]")
    
    
    
    def _handle_hold_start(self):
        """Handle tap and hold start (confirmation)."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        finger_count = len(self.fingers)
        positions = [(fx, fy) for _, _, fx, fy, _ in self.fingers.values()]
        
        # Calculate time until hold confirmation
        hold_start_time = self.gesture_hold_state['start_time']
        confirmation_time = time.time()
        time_to_confirm = (confirmation_time - hold_start_time) * 1000
        
        # Store start positions for end event
        self.gesture_hold_state['start_positions'] = positions
        
        print(f"[{timestamp}] 🤚 TAP AND HOLD CONFIRMED: {finger_count} finger(s)")
        print(f"   Time to confirm: {time_to_confirm:.0f}ms")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] 🤚 TAP AND HOLD CONFIRMED: {finger_count} fingers\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def _handle_hold_end(self):
        """Handle tap and hold end."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        finger_count = len(self.fingers)
        positions = [(fx, fy) for _, _, fx, fy, _ in self.fingers.values()]
        
        # Calculate hold duration from actual start time
        hold_start_time = self.gesture_hold_state['start_time']
        hold_end_time = time.time()
        hold_duration = hold_end_time - hold_start_time
        
        # Format start time for display
        start_timestamp = datetime.datetime.fromtimestamp(hold_start_time).strftime("%H:%M:%S.%f")[:-3]
        
        print(f"[{timestamp}] ✋ TAP AND HOLD END: {finger_count} finger(s)")
        print(f"   Started at: {start_timestamp}")
        print(f"   Total duration: {hold_duration:.1f}s")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] ✋ TAP AND HOLD END: {finger_count} fingers, duration: {hold_duration:.1f}s\n\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def _track_motion(self, slot, x, y):
        """Track finger motion for scrub detection."""
        if slot not in self.motion_history:
            self.motion_history[slot] = {
                'positions': [],
                'directions': [],
                'total_distance': 0
            }
        
        history = self.motion_history[slot]
        
        # Add new position
        history['positions'].append((x, y))
        
        # Keep only recent positions (last 20)
        if len(history['positions']) > 20:
            history['positions'] = history['positions'][-20:]
        
        # Calculate direction changes if we have enough positions
        if len(history['positions']) >= 3:
            # Get last 3 positions to detect direction
            p1, p2, p3 = history['positions'][-3:]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate directions
            dir1 = math.degrees(math.atan2(-v1[1], v1[0]))
            dir2 = math.degrees(math.atan2(-v2[1], v2[0]))
            
            if dir1 < 0: dir1 += 360
            if dir2 < 0: dir2 += 360
            
            # Check for direction change (more than 90 degrees)
            angle_diff = abs(dir1 - dir2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > 90:
                history['directions'].append((dir1, dir2))
                
                # Keep only recent direction changes
                if len(history['directions']) > 10:
                    history['directions'] = history['directions'][-10:]
        
        # Calculate total distance traveled
        if len(history['positions']) >= 2:
            last_pos = history['positions'][-2]
            distance = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
            history['total_distance'] += distance
    
    def _check_scrub_motion(self, slot):
        """Check if the motion history indicates a scrub gesture."""
        if slot not in self.motion_history:
            return False
        
        history = self.motion_history[slot]
        
        # Check minimum requirements
        if (len(history['directions']) >= self.SCRUB_MIN_DIRECTION_CHANGES and
            history['total_distance'] >= self.SCRUB_MIN_TOTAL_DISTANCE):
            return True
        
        return False
    
    def _handle_scrub_gesture(self, finger_count, positions):
        """Handle scrub gesture detection."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get motion statistics
        slot = list(self.motion_history.keys())[0]  # Use first slot
        history = self.motion_history[slot]
        
        print(f"[{timestamp}] 🔄 SCRUB GESTURE: {finger_count} finger(s)")
        print(f"   Direction changes: {len(history['directions'])}")
        print(f"   Total distance: {int(history['total_distance'])}px")
        
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(x)}, {int(y)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] 🔄 SCRUB GESTURE: {finger_count} fingers, changes: {len(history['directions'])}, distance: {int(history['total_distance'])}px\n\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def _check_directional_swipe(self, sx, sy, ex, ey, distance):
        """Check if swipe is a directional swipe from/to center."""
        if distance < self.DIRECTIONAL_SWIPE_MIN_DISTANCE:
            return None
        
        # Calculate swipe angle (0° = East, 90° = North, 180° = West, 270° = South)
        angle = math.degrees(math.atan2(-(ey - sy), ex - sx))
        if angle < 0:
            angle += 360
        
        # Debug output for FROM_CENTER detection
        center_distance = math.sqrt((sx - self.center_x)**2 + (sy - self.center_y)**2)
        center_threshold = min(self.screen_width, self.screen_height) * 0.1
        
        if center_distance < center_threshold:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            # FROM_CENTER DEBUG: start=({sx},{sy}), angle={angle:.1f}°
        
        # Check if swipe starts near a compass point and heads toward center
        for compass_point in self.compass_points:
            target_x, target_y = self.compass_regions[compass_point]
            
            # Check if start is near this compass point
            start_distance = math.sqrt((sx - target_x)**2 + (sy - target_y)**2)
            start_threshold = min(self.screen_width, self.screen_height) * 0.15  # 15% of smaller dimension
            
            if start_distance < start_threshold:
                # Check if swipe direction is toward center
                center_angle = math.degrees(math.atan2(-(self.center_y - sy), self.center_x - sx))
                if center_angle < 0:
                    center_angle += 360
                
                angle_diff = abs(angle - center_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff <= self.DIRECTIONAL_SWIPE_ANGLE_TOLERANCE:
                    return ("TO_CENTER", compass_point)
            
            # Check if swipe starts near center and heads toward compass point
            if center_distance < center_threshold:
                # Check if swipe direction is toward compass point
                target_angle = math.degrees(math.atan2(-(target_y - sy), target_x - sx))
                if target_angle < 0:
                    target_angle += 360
                
                angle_diff = abs(angle - target_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Debug output for corner detection
                # if compass_point in ['NE', 'SE', 'SW', 'NW']:  # Corner points
                #     import datetime
                #     timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                #     print(f"[{timestamp}] 🔍 CORNER {compass_point}: target_angle={target_angle:.1f}°, swipe_angle={angle:.1f}°, diff={angle_diff:.1f}° (tolerance={self.DIRECTIONAL_SWIPE_ANGLE_TOLERANCE}°)")
                
                if angle_diff <= self.DIRECTIONAL_SWIPE_ANGLE_TOLERANCE:
                    return ("FROM_CENTER", compass_point)
        
        return None


def main():
    listener = TouchListener()
    
    if not listener.start():
        return
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()