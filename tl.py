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
        self.active_slots = set()
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
        self.TAP_HOLD_TIMEOUT = 500  # 500ms (0.5 seconds) before recognizing as hold
        self.TAP_HOLD_MAX_MOVEMENT_PERCENT = 0.5  # Reduced to 0.5% for stricter hold detection and better scrub distinction
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
        self.SCRUB_MIN_SEGMENTS = 4  # Minimum vertical segments for scrub
        self.SCRUB_MIN_SEGMENT_PERCENT = 3.0  # Percent of screen height for each segment
        self.SCRUB_MIN_VERTICAL_PERCENT = 20.0  # Percent of screen height for total vertical travel
        # Pinch detection parameters
        self.PINCH_MIN_FINGERS = 2
        self.PINCH_MAX_FINGERS = 5  # Support up to 5 fingers
        self.PINCH_MIN_CHANGE_PERCENT = 20.0  # Minimum percentage change in distance for pinch
        self.PINCH_DIRECTION_ANGLE_TOLERANCE = 45  # Degrees; tolerance for opposing directions (e.g., 180° ±45° for pinch)
        self.pinch_data = {
            'initial_distance': 0.0,
            'min_distance': float('inf'),
            'max_distance': 0.0,
            'is_active': False
        }
        
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
        self.SCRUB_MIN_SEGMENT_DISTANCE = int(self.screen_height * self.SCRUB_MIN_SEGMENT_PERCENT / 100)
        self.SCRUB_MIN_VERTICAL_TRAVEL = int(self.screen_height * self.SCRUB_MIN_VERTICAL_PERCENT / 100)
        
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
            
            event_batch = []
            for event in self.device.read_loop():
                if not self.running:
                    break
                
                event_batch.append(event)
                
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                    # Process the batch of events
                    for ev in event_batch:
                        if ev.type == ecodes.EV_ABS:
                            if ev.code == ecodes.ABS_MT_SLOT:
                                current_slot = ev.value
                            elif ev.code == ecodes.ABS_MT_TRACKING_ID:
                                slot = current_slot
                                if ev.value == -1:
                                    # Finger lifted
                                    if slot in self.fingers:
                                        # Remove from slot-level hold tracking
                                        if slot in self.active_holds:
                                            del self.active_holds[slot]
                                        
                                        if slot in self.active_slots:
                                            self.active_slots.remove(slot)
                                        
                                        if not self.active_slots:
                                            print(f"🔍 DEBUG: All fingers lifted, active_slots empty, is_hold={self.gesture_hold_state['is_hold']}")
                                            if self.gesture_hold_state['is_hold']:
                                                self._handle_hold_end()
                                            else:
                                                self._process_gesture()
                                            self.fingers.clear()
                                            self.active_slots.clear()
                                            # Reset gesture hold state
                                            self.gesture_hold_state = {
                                                'is_hold': False,
                                                'start_time': 0,
                                                'notified': False,
                                                'start_positions': []
                                            }
                                            self.pinch_data = {
                                                'initial_distance': 0.0,
                                                'min_distance': float('inf'),
                                                'max_distance': 0.0,
                                                'is_active': False
                                            }
                                else:
                                    # Finger placed
                                    slot_data[slot] = {'x': 0, 'y': 0}  # Always reset for new touch
                                    self.fingers[slot] = (0, 0, 0, 0, time.time())
                                    self.active_slots.add(slot)
                                    
                                    # Initialize gesture hold state if this is the first finger
                                    if len(self.active_slots) == 1:
                                        self.gesture_hold_state = {
                                            'is_hold': False,
                                            'start_time': time.time(),
                                            'notified': False,
                                            'start_positions': []
                                        }
                                        # First finger touched, initializing hold state
                                    else:
                                        print(f"🔍 Additional finger touched, total fingers: {len(self.active_slots)}")
                                    
                                    # Start tracking for potential hold (per slot)
                                    self.active_holds[slot] = {
                                        'start_time': time.time(),
                                        'start_x': 0,
                                        'start_y': 0,
                                        'notified': False
                                    }
                            elif ev.code == ecodes.ABS_MT_POSITION_X:
                                slot = current_slot
                                if slot in slot_data:
                                    slot_data[slot]['x'] = ev.value
                                    if slot in self.fingers:
                                        sx, sy, _, _, st = self.fingers[slot]
                                        if sx == 0:
                                            current_y = slot_data[slot].get('y', 0)
                                            self.fingers[slot] = (ev.value, current_y, ev.value, current_y, st)
                                            print(f"🔍 DEBUG: Initial X for slot {slot}: start=({ev.value}, {current_y}), end= same")
                                        else:
                                            new_ex = ev.value
                                            new_ey = slot_data[slot]['y']
                                            self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                                            self._track_motion(slot, new_ex, new_ey)
                                            print(f"🔍 DEBUG: Updated X for slot {slot}: start=({sx}, {sy}), end=({new_ex}, {new_ey})")
                                    if slot in self.active_holds:
                                        if self.active_holds[slot]['start_x'] == 0:
                                            self.active_holds[slot]['start_x'] = ev.value
                                        else:
                                            movement = abs(ev.value - self.active_holds[slot]['start_x'])
                                            if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                                del self.active_holds[slot]
                                                print(f"🔍 DEBUG: Removed hold tracking for slot {slot} due to X movement: {movement} > {self.TAP_HOLD_MAX_MOVEMENT}")
                            elif ev.code == ecodes.ABS_MT_POSITION_Y:
                                slot = current_slot
                                if slot in slot_data:
                                    slot_data[slot]['y'] = ev.value
                                    if slot in self.fingers:
                                        sx, sy, _, _, st = self.fingers[slot]
                                        if sy == 0:
                                            current_x = slot_data[slot].get('x', 0)
                                            self.fingers[slot] = (current_x, ev.value, current_x, ev.value, st)
                                            print(f"🔍 DEBUG: Initial Y for slot {slot}: start=({current_x}, {ev.value}), end= same")
                                        else:
                                            new_ex = slot_data[slot]['x']
                                            new_ey = ev.value
                                            self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                                            self._track_motion(slot, new_ex, new_ey)
                                            print(f"🔍 DEBUG: Updated Y for slot {slot}: start=({sx}, {sy}), end=({new_ex}, {new_ey})")
                                    if slot in self.active_holds:
                                        if self.active_holds[slot]['start_y'] == 0:
                                            self.active_holds[slot]['start_y'] = ev.value
                                        else:
                                            movement = abs(ev.value - self.active_holds[slot]['start_y'])
                                            if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                                del self.active_holds[slot]
                                                print(f"🔍 DEBUG: Removed hold tracking for slot {slot} due to Y movement: {movement} > {self.TAP_HOLD_MAX_MOVEMENT}")
                    
                    # After processing batch, check for hold
                    current_time = time.time()
                    if self.gesture_hold_state['start_time'] > 0 and not self.gesture_hold_state['notified']:
                        hold_duration = current_time - self.gesture_hold_state['start_time']
                        hold_duration_ms = hold_duration * 1000
                        
                        effective_hold_timeout = self.TAP_HOLD_TIMEOUT if len(self.active_slots) < 2 or len(self.active_slots) > self.PINCH_MAX_FINGERS else self.TAP_HOLD_TIMEOUT * 0.8  # 20% shorter for 2 fingers to avoid slow pinch conflicts
                        if hold_duration_ms >= effective_hold_timeout:
                            all_fingers_stable = True
                            for slot in list(self.active_slots):
                                if slot in self.fingers:
                                    sx, sy, ex, ey, st = self.fingers[slot]
                                    movement = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                                    if movement > self.TAP_HOLD_MAX_MOVEMENT:
                                        all_fingers_stable = False
                                        break
                            
                            if all_fingers_stable:
                                self.gesture_hold_state['is_hold'] = True
                                self.gesture_hold_state['notified'] = True
                                self._handle_hold_start()
                                print(f"🔍 DEBUG: Hold confirmed, duration={hold_duration_ms:.0f}ms, stable={all_fingers_stable}")
                            else:
                                self.gesture_hold_state['is_hold'] = False
                                print(f"🔍 DEBUG: Hold rejected due to movement, duration={hold_duration_ms:.0f}ms, stable={all_fingers_stable}")
                    
                    # Track pinch distance in real-time
                    if len(self.active_slots) >= self.PINCH_MIN_FINGERS and len(self.active_slots) <= self.PINCH_MAX_FINGERS and not self.pinch_data['is_active']:
                        current_distance = self._calculate_distance()
                        if current_distance > 0:
                            slots = list(self.active_slots)
                            self.pinch_data['initial_distance'] = current_distance
                            self.pinch_data['min_distance'] = current_distance
                            self.pinch_data['max_distance'] = current_distance
                            self.pinch_data['is_active'] = True
                            finger_desc = "fingers" if len(slots) > 2 else "finger"
                            print(f"🔍 DEBUG: Pinch tracking started - {len(slots)} {finger_desc}, initial_distance={current_distance:.1f}")
                    elif (len(self.active_slots) < self.PINCH_MIN_FINGERS or len(self.active_slots) > self.PINCH_MAX_FINGERS) and self.pinch_data['is_active']:
                        self.pinch_data['is_active'] = False
                        print("🔍 DEBUG: Pinch tracking stopped due to finger count change")
                    if len(self.active_slots) >= self.PINCH_MIN_FINGERS and len(self.active_slots) <= self.PINCH_MAX_FINGERS and self.pinch_data['is_active']:
                        current_distance = self._calculate_distance()
                        if current_distance > 0:
                            old_min = self.pinch_data['min_distance']
                            old_max = self.pinch_data['max_distance']
                            self.pinch_data['min_distance'] = min(old_min, current_distance)
                            self.pinch_data['max_distance'] = max(old_max, current_distance)
                            if self.pinch_data['min_distance'] != old_min or self.pinch_data['max_distance'] != old_max:
                                print(f"🔍 DEBUG: Distance updated - current={current_distance:.1f}, min={self.pinch_data['min_distance']:.1f}, max={self.pinch_data['max_distance']:.1f}")
                    
                    # Clear batch
                    event_batch = []
            for event in self.device.read_loop():
                if not self.running:
                    break
                
                event_batch.append(event)
                
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                    # Process the batch of events
                    for ev in event_batch:
                        if ev.type == ecodes.EV_ABS:
                            if ev.code == ecodes.ABS_MT_SLOT:
                                current_slot = ev.value
                            elif ev.code == ecodes.ABS_MT_TRACKING_ID:
                                slot = current_slot
                                if ev.value == -1:
                                    # Finger lifted
                                    if slot in self.fingers:
                                        # Remove from slot-level hold tracking
                                        if slot in self.active_holds:
                                            del self.active_holds[slot]
                                        
                                        if slot in self.active_slots:
                                            self.active_slots.remove(slot)
                                        
                                        if not self.active_slots:
                                            print(f"🔍 DEBUG: All fingers lifted, active_slots empty, is_hold={self.gesture_hold_state['is_hold']}")
                                            if self.gesture_hold_state['is_hold']:
                                                self._handle_hold_end()
                                            else:
                                                self._process_gesture()
                                            self.fingers.clear()
                                            self.active_slots.clear()
                                            # Reset gesture hold state
                                            self.gesture_hold_state = {
                                                'is_hold': False,
                                                'start_time': 0,
                                                'notified': False,
                                                'start_positions': []
                                            }
                                            self.pinch_data = {
                                                'initial_distance': 0.0,
                                                'min_distance': float('inf'),
                                                'max_distance': 0.0,
                                                'is_active': False
                                            }
                                else:
                                    # Finger placed
                                    slot_data[slot] = {'x': 0, 'y': 0}  # Always reset for new touch
                                    self.fingers[slot] = (0, 0, 0, 0, time.time())
                                    self.active_slots.add(slot)
                                    
                                    # Initialize gesture hold state if this is the first finger
                                    if len(self.active_slots) == 1:
                                        self.gesture_hold_state = {
                                            'is_hold': False,
                                            'start_time': time.time(),
                                            'notified': False,
                                            'start_positions': []
                                        }
                                        # First finger touched, initializing hold state
                                    else:
                                        print(f"🔍 Additional finger touched, total fingers: {len(self.active_slots)}")
                                    
                                    # Start tracking for potential hold (per slot)
                                    self.active_holds[slot] = {
                                        'start_time': time.time(),
                                        'start_x': 0,
                                        'start_y': 0,
                                        'notified': False
                                    }
                            elif ev.code == ecodes.ABS_MT_POSITION_X:
                                slot = current_slot
                                if slot in slot_data:
                                    slot_data[slot]['x'] = ev.value
                                    if slot in self.fingers:
                                        sx, sy, _, _, st = self.fingers[slot]
                                        if sx == 0:
                                            current_y = slot_data[slot].get('y', 0)
                                            self.fingers[slot] = (ev.value, current_y, ev.value, current_y, st)
                                            print(f"🔍 DEBUG: Initial X for slot {slot}: start=({ev.value}, {current_y}), end= same")
                                        else:
                                            new_ex = ev.value
                                            new_ey = slot_data[slot]['y']
                                            self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                                            self._track_motion(slot, new_ex, new_ey)
                                            print(f"🔍 DEBUG: Updated X for slot {slot}: start=({sx}, {sy}), end=({new_ex}, {new_ey})")
                                    if slot in self.active_holds:
                                        if self.active_holds[slot]['start_x'] == 0:
                                            self.active_holds[slot]['start_x'] = ev.value
                                        else:
                                            movement = abs(ev.value - self.active_holds[slot]['start_x'])
                                            if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                                del self.active_holds[slot]
                                                print(f"🔍 DEBUG: Removed hold tracking for slot {slot} due to X movement: {movement} > {self.TAP_HOLD_MAX_MOVEMENT}")
                            elif ev.code == ecodes.ABS_MT_POSITION_Y:
                                slot = current_slot
                                if slot in slot_data:
                                    slot_data[slot]['y'] = ev.value
                                    if slot in self.fingers:
                                        sx, sy, _, _, st = self.fingers[slot]
                                        if sy == 0:
                                            current_x = slot_data[slot].get('x', 0)
                                            self.fingers[slot] = (current_x, ev.value, current_x, ev.value, st)
                                            print(f"🔍 DEBUG: Initial Y for slot {slot}: start=({current_x}, {ev.value}), end= same")
                                        else:
                                            new_ex = slot_data[slot]['x']
                                            new_ey = ev.value
                                            self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                                            self._track_motion(slot, new_ex, new_ey)
                                            print(f"🔍 DEBUG: Updated Y for slot {slot}: start=({sx}, {sy}), end=({new_ex}, {new_ey})")
                                    if slot in self.active_holds:
                                        if self.active_holds[slot]['start_y'] == 0:
                                            self.active_holds[slot]['start_y'] = ev.value
                                        else:
                                            movement = abs(ev.value - self.active_holds[slot]['start_y'])
                                            if movement > self.TAP_HOLD_MAX_MOVEMENT and not self.active_holds[slot]['notified']:
                                                del self.active_holds[slot]
                                                print(f"🔍 DEBUG: Removed hold tracking for slot {slot} due to Y movement: {movement} > {self.TAP_HOLD_MAX_MOVEMENT}")
                    
                    # After processing batch, check for hold
                    current_time = time.time()
                    if self.gesture_hold_state['start_time'] > 0 and not self.gesture_hold_state['notified']:
                        hold_duration = current_time - self.gesture_hold_state['start_time']
                        hold_duration_ms = hold_duration * 1000
                        
                        effective_hold_timeout = self.TAP_HOLD_TIMEOUT if len(self.active_slots) < 2 or len(self.active_slots) > self.PINCH_MAX_FINGERS else self.TAP_HOLD_TIMEOUT * 0.8  # 20% shorter for 2 fingers to avoid slow pinch conflicts
                        if hold_duration_ms >= effective_hold_timeout:
                            all_fingers_stable = True
                            for slot in list(self.active_slots):
                                if slot in self.fingers:
                                    sx, sy, ex, ey, st = self.fingers[slot]
                                    movement = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                                    if movement > self.TAP_HOLD_MAX_MOVEMENT:
                                        all_fingers_stable = False
                                        break
                            
                            if all_fingers_stable:
                                self.gesture_hold_state['is_hold'] = True
                                self.gesture_hold_state['notified'] = True
                                self._handle_hold_start()
                                print(f"🔍 DEBUG: Hold confirmed, duration={hold_duration_ms:.0f}ms, stable={all_fingers_stable}")
                            else:
                                self.gesture_hold_state['is_hold'] = False
                                print(f"🔍 DEBUG: Hold rejected due to movement, duration={hold_duration_ms:.0f}ms, stable={all_fingers_stable}")
                    
                    # Track pinch distance in real-time
                    if len(self.active_slots) >= self.PINCH_MIN_FINGERS and len(self.active_slots) <= self.PINCH_MAX_FINGERS and not self.pinch_data['is_active']:
                        current_distance = self._calculate_distance()
                        if current_distance > 0:
                            slots = list(self.active_slots)
                            self.pinch_data['initial_distance'] = current_distance
                            self.pinch_data['min_distance'] = current_distance
                            self.pinch_data['max_distance'] = current_distance
                            self.pinch_data['is_active'] = True
                            finger_desc = "fingers" if len(slots) > 2 else "finger"
                            print(f"🔍 DEBUG: Pinch tracking started - {len(slots)} {finger_desc}, initial_distance={current_distance:.1f}")
                    elif (len(self.active_slots) < self.PINCH_MIN_FINGERS or len(self.active_slots) > self.PINCH_MAX_FINGERS) and self.pinch_data['is_active']:
                        self.pinch_data['is_active'] = False
                        print("🔍 DEBUG: Pinch tracking stopped due to finger count change")
                    if len(self.active_slots) >= self.PINCH_MIN_FINGERS and len(self.active_slots) <= self.PINCH_MAX_FINGERS and self.pinch_data['is_active']:
                        current_distance = self._calculate_distance()
                        if current_distance > 0:
                            old_min = self.pinch_data['min_distance']
                            old_max = self.pinch_data['max_distance']
                            self.pinch_data['min_distance'] = min(old_min, current_distance)
                            self.pinch_data['max_distance'] = max(old_max, current_distance)
                            if self.pinch_data['min_distance'] != old_min or self.pinch_data['max_distance'] != old_max:
                                print(f"🔍 DEBUG: Distance updated - current={current_distance:.1f}, min={self.pinch_data['min_distance']:.1f}, max={self.pinch_data['max_distance']:.1f}")
                    
                    # Clear batch
                    event_batch = []        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _process_gesture(self):
        if not self.fingers:
            return
        
        finger_count = len(self.fingers)
        positions = [(x, y) for _, _, x, y, _ in self.fingers.values()]
        
        # Calculate movement
        max_move = 0
        for sx, sy, ex, ey, _ in self.fingers.values():
            move = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            max_move = max(max_move, move)
        
            # Check for pinch with vector checks (supports 2-5 fingers)
        is_pinch = False
        pinch_type = None
        final_distance = 0.0
        pinch_zoom_direction = None
        
        if finger_count >= self.PINCH_MIN_FINGERS and finger_count <= self.PINCH_MAX_FINGERS and self.pinch_data['initial_distance'] > 0:
            slots = list(self.fingers.keys())
            
            if finger_count == 2:
                # 2-finger pinch: check if vectors are opposing
                sx1, sy1, ex1, ey1, _ = self.fingers[slots[0]]
                sx2, sy2, ex2, ey2, _ = self.fingers[slots[1]]
                vec1 = (ex1 - sx1, ey1 - sy1)
                vec2 = (ex2 - sx2, ey2 - sy2)
                
                # Compute angles
                angle1 = math.degrees(math.atan2(vec1[1], vec1[0]))
                angle2 = math.degrees(math.atan2(vec2[1], vec2[0]))
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Check if directions are opposing (for pinch)
                directions_opposing = abs(angle_diff - 180) <= self.PINCH_DIRECTION_ANGLE_TOLERANCE
                
                # Compute distance change
                final_distance = math.sqrt((ex1 - ex2)**2 + (ey1 - ey2)**2)
                pinch_zoom_direction = None  # Not used for 2-finger
                
            elif finger_count >= 3:
                # 3-5 finger pinch: check if all fingers are moving toward/away from center
                positions = [(self.fingers[slot][2], self.fingers[slot][3]) for slot in slots]
                start_positions = [(self.fingers[slot][0], self.fingers[slot][1]) for slot in slots]
                
                # Calculate center of polygon
                center_x = sum(x for x, y in positions) / finger_count
                center_y = sum(y for x, y in positions) / finger_count
                start_center_x = sum(x for x, y in start_positions) / finger_count
                start_center_y = sum(y for x, y in start_positions) / finger_count
                
                # Check if all fingers are moving toward or away from center
                all_converging = True
                all_diverging = True
                
                # Calculate the average distance from center for start and end positions
                start_avg_distance = 0
                end_avg_distance = 0
                
                for i, (sx, sy, ex, ey, _) in enumerate([self.fingers[slot] for slot in slots]):
                    # Vector from start to end position
                    dx = ex - sx
                    dy = ey - sy
                    
                    # Vector from start position to center
                    start_vec_x = start_center_x - start_positions[i][0]
                    start_vec_y = start_center_y - start_positions[i][1]
                    
                    # Check dot product to see if moving toward/away from center
                    if dx * start_vec_x + dy * start_vec_y < 0:
                        all_converging = False
                    if dx * start_vec_x + dy * start_vec_y > 0:
                        all_diverging = False
                    
                    # Calculate distances from center
                    start_avg_distance += math.sqrt((start_positions[i][0] - start_center_x)**2 + 
                                                 (start_positions[i][1] - start_center_y)**2)
                    end_avg_distance += math.sqrt((positions[i][0] - center_x)**2 + 
                                               (positions[i][1] - center_y)**2)
                
                start_avg_distance /= finger_count
                end_avg_distance /= finger_count
                
                directions_opposing = all_converging or all_diverging
                final_distance = self._calculate_distance()
                
                # Use the average distance from center to determine zoom direction
                # This is more reliable than the pairwise distance for 3+ fingers
                if end_avg_distance > start_avg_distance:
                    pinch_zoom_direction = 'ZOOM_OUT'
                else:
                    pinch_zoom_direction = 'ZOOM_IN'            
            distance_change = abs(self.pinch_data['max_distance'] - self.pinch_data['min_distance'])
            min_change = self.pinch_data['initial_distance'] * (self.PINCH_MIN_CHANGE_PERCENT / 100)
            
            if distance_change >= min_change and directions_opposing:
                is_pinch = True
                if finger_count >= 3:
                    # Use the more reliable direction determined from center distances
                    pinch_type = pinch_zoom_direction
                else:
                    # 2-finger pinch uses distance comparison
                    if final_distance > self.pinch_data['initial_distance']:
                        pinch_type = 'ZOOM_IN'
                    else:
                        pinch_type = 'ZOOM_OUT'
        
        # Check for scrub motion - check all slots
        is_scrub = False
        for slot in self.fingers.keys():
            if self._check_scrub_motion(slot):
                is_scrub = True
                break
        
        # Gesture classification priority: scrub > pinch > tap > swipe
        if is_scrub:
            self._handle_scrub_gesture(finger_count, positions)
        elif is_pinch:
            self._handle_pinch(pinch_type, finger_count, positions, final_distance)
        elif max_move < self.TAP_DISTANCE:
            self._handle_tap(finger_count, positions)
        else:
            self._handle_swipe(finger_count, positions)
        
        # Cleanup motion history for all slots
        for slot in list(self.motion_history.keys()):
            if slot in self.motion_history:
                del self.motion_history[slot]
    
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
        
        print(f"[{timestamp}] 🤚 TOUCH AND HOLD CONFIRMED: {finger_count} finger(s)")
        print(f"   Time to confirm: {time_to_confirm:.0f}ms")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] 🤚 TOUCH AND HOLD CONFIRMED: {finger_count} fingers\n"
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
        
        print(f"[{timestamp}] ✋ TOUCH AND HOLD END: {finger_count} finger(s)")
        print(f"   Started at: {start_timestamp}")
        print(f"   Total duration: {hold_duration:.1f}s")
        
        for i, (fx, fy) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(fx)}, {int(fy)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] ✋ TOUCH AND HOLD END: {finger_count} fingers, duration: {hold_duration:.1f}s\n\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def _track_motion(self, slot, x, y):
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
                dir = 'down' if history['segment_dy'] > 0 else 'up'
                history['vertical_segments'].append(dir)
                history['segment_dy'] = dy
        history['last_x'] = x
        history['last_y'] = y
    
    def _check_scrub_motion(self, slot):
        """Check if the motion history indicates a scrub gesture."""
        if slot not in self.motion_history:
            return False
        
        history = self.motion_history[slot]
        
        num_segments = len(history['vertical_segments'])
        if abs(history['segment_dy']) >= self.SCRUB_MIN_SEGMENT_DISTANCE:
            num_segments += 1
        
        if (num_segments >= self.SCRUB_MIN_SEGMENTS and
            history['total_vertical_travel'] >= self.SCRUB_MIN_VERTICAL_TRAVEL and
            history['total_vertical_travel'] > history['total_horizontal_travel'] * 1.5):
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
        print(f"   Vertical segments: {len(history['vertical_segments'])}")
        print(f"   Total vertical travel: {int(history['total_vertical_travel'])}px")
        
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(x)}, {int(y)})")
        
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] 🔄 SCRUB GESTURE: {finger_count} fingers, segments: {len(history['vertical_segments'])}, vertical travel: {int(history['total_vertical_travel'])}px\n\n"
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


    def _handle_pinch(self, pinch_type, finger_count, positions, final_distance):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] 🔍 PINCH {pinch_type}: {finger_count} finger(s) (directions opposing)")
        print(f"   Initial distance: {self.pinch_data['initial_distance']:.1f}px, Final distance: {final_distance:.1f}px")
        for i, (x, y) in enumerate(positions):
            print(f"   Finger {i+1}: ({int(x)}, {int(y)})")
        change_percent = ((self.pinch_data['max_distance'] - self.pinch_data['min_distance']) / self.pinch_data['initial_distance']) * 100 if self.pinch_data['initial_distance'] > 0 else 0
        print(f"   Distance change: {change_percent:.1f}%")
        if self.debug_file:
            try:
                debug_msg = f"[{timestamp}] 🔍 PINCH {pinch_type}: {finger_count} fingers, change={change_percent:.1f}%, initial={self.pinch_data['initial_distance']:.1f}, final={final_distance:.1f}\n\n"
                self.debug_file.write(debug_msg)
                self.debug_file.flush()
            except:
                pass
    
    def _calculate_distance(self):
        finger_count = len(self.active_slots)
        if finger_count < 2 or finger_count > self.PINCH_MAX_FINGERS:
            return 0.0
        
        slots = list(self.active_slots)
        if finger_count == 2:
            # 2-finger pinch: simple distance between two points
            x1, y1 = self.fingers[slots[0]][2], self.fingers[slots[0]][3]
            x2, y2 = self.fingers[slots[1]][2], self.fingers[slots[1]][3]
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif finger_count >= 3:
            # 3-5 finger pinch: use average distance between all pairs
            positions = [(self.fingers[slot][2], self.fingers[slot][3]) for slot in slots]
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    distances.append(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
            return sum(distances) / len(distances)  # Average distance
        return 0.0

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