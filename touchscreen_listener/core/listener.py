"""
Main touchscreen listener class that coordinates device management and gesture detection.
"""

import time
import threading
import queue
import logging
import math
from typing import Dict, Tuple, Optional
from evdev import ecodes

from ..device.device_manager import DeviceManager
from ..gestures.gesture_detector import GestureDetector
from ..utils.logger import TouchLogger

class TouchListener:
    """Main touchscreen listener that coordinates device management and gesture detection."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.gesture_detector = None
        self.logger = TouchLogger()
        
        # State management
        self.running = False
        self.fingers = {}
        self.active_slots = set()
        self.current_slot = 0
        self.slot_data = {}
        
        # Complete finger path storage for Douglas-Peucker algorithm
        self.finger_paths = {}  # slot -> list of {'x', 'y', 't', 'slot'}
        
        # Hold and drag tracking
        self.active_holds = {}
        self.gesture_hold_state = {
            'is_hold': False,
            'start_time': 0,
            'notified': False,
            'start_positions': []
        }
        
        self.gesture_drag_hold_state = {
            'is_drag_hold': False,
            'drag_start_time': 0,
            'stop_start_time': 0,
            'notified': False,
            'initial_drag_detected': False,
            'start_positions': [],
            'drag_end_positions': []
        }
        
        self.motion_stop_tracking = {}
        
        # Thread management
        self.thread = None
        self.hold_thread = None
        self.state_lock = threading.Lock()
    
    def start(self) -> bool:
        """Start the touchscreen listener."""
        device = self.device_manager.find_device()
        if not device:
            print("❌ No touchscreen found")
            return False
        
        # Initialize gesture detector with screen dimensions
        device_info = self.device_manager.get_device_info()
        self.gesture_detector = GestureDetector(
            device_info['screen_width'],
            device_info['screen_height']
        )
        
        self.running = True
        self._print_startup_info(device_info)
        
        self.thread = threading.Thread(target=self._event_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.hold_thread = threading.Thread(target=self._hold_checker)
        self.hold_thread.daemon = True
        self.hold_thread.start()
        
        return True
    
    def stop(self):
        """Stop the touchscreen listener."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.hold_thread:
            self.hold_thread.join(timeout=1)
        self.logger.close()
    
    def _print_startup_info(self, device_info: Dict):
        """Print startup information."""
        print(f"✅ Found: {self.device_manager.device.name}")
        print(f"📺 Screen: {device_info['screen_width']}x{device_info['screen_height']}")
        print(f"📏 Tap threshold: {self.gesture_detector.TAP_DISTANCE}px")
        print(f"📏 Swipe threshold: {self.gesture_detector.SWIPE_DISTANCE}px")
        print(f"📏 Double tap distance: {self.gesture_detector.DOUBLE_TAP_MAX_DISTANCE}px")
        print(f"📏 Directional swipe min distance: {self.gesture_detector.DIRECTIONAL_SWIPE_MIN_DISTANCE}px")
        print(f"📏 Drag hold min drag: {self.gesture_detector.DRAG_HOLD_MIN_DRAG_DISTANCE}px")
        print(f"📏 Drag hold stop movement: {self.gesture_detector.DRAG_HOLD_STOP_MOVEMENT}px")
        print(f"🎯 Screen center: ({device_info['center_x']}, {device_info['center_y']})")
        print("🎯 Ready! Try taps, swipes, and directional swipes from corners/edges to center!")
    
    def _event_loop(self):
        """Main event processing loop."""
        try:
            event_batch = []
            for event in self.device_manager.device.read_loop():
                if not self.running:
                    break
                
                event_batch.append(event)
                
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                    with self.state_lock:
                        self._process_event_batch(event_batch)
                    event_batch = []
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error(f"Error in event loop: {e}")
    
    def _process_event_batch(self, event_batch):
        """Process a batch of events."""
        for ev in event_batch:
            if ev.type == ecodes.EV_ABS:
                self._handle_abs_event(ev)
        
        # After processing batch, check for gestures and holds
        self._check_gesture_state()
    
    def _handle_abs_event(self, ev):
        """Handle absolute coordinate events."""
        if ev.code == ecodes.ABS_MT_SLOT:
            self.current_slot = ev.value
        elif ev.code == ecodes.ABS_MT_TRACKING_ID:
            self._handle_tracking_id(ev.value)
        elif ev.code == ecodes.ABS_MT_POSITION_X:
            self._handle_position_x(ev.value)
        elif ev.code == ecodes.ABS_MT_POSITION_Y:
            self._handle_position_y(ev.value)
    
    def _handle_tracking_id(self, value: int):
        """Handle finger tracking ID changes."""
        slot = self.current_slot
        
        if value == -1:
            # Finger lifted
            self._handle_finger_lift(slot)
        else:
            # Finger placed
            self._handle_finger_place(slot)
    
    def _handle_finger_lift(self, slot: int):
        """Handle finger lift event."""
        if slot in self.fingers:
            # Clean up tracking
            self.active_slots.discard(slot)
            self.active_holds.pop(slot, None)
            self.motion_stop_tracking.pop(slot, None)
            
            # Clean up path storage for this finger
            if slot in self.finger_paths:
                del self.finger_paths[slot]
            
            if not self.active_slots:
                # Check if we had a drag and hold that was confirmed
                if self.gesture_drag_hold_state['is_drag_hold']:
                    finger_count = len(self.fingers)
                    positions = [
                        (ex, ey) for _, _, ex, ey, _ in self.fingers.values()
                    ]
                    self.logger.log_drag_hold_end(
                        finger_count,
                        positions,
                        self.gesture_drag_hold_state['drag_start_time'],
                        self.gesture_drag_hold_state['stop_start_time']
                    )
                else:
                    # Only process other gestures if not drag hold
                    self._process_gesture()
                self._reset_gesture_state()
    
    def _handle_finger_place(self, slot: int):
        """Handle finger placement event."""
        self.slot_data[slot] = {'x': 0, 'y': 0}
        self.fingers[slot] = (0, 0, 0, 0, time.time())
        self.active_slots.add(slot)
        
        # Initialize path storage for this finger
        self.finger_paths[slot] = []
        
        # Initialize gesture hold state if first finger
        if len(self.active_slots) == 1:
            self.gesture_hold_state = {
                'is_hold': False,
                'start_time': time.time(),
                'notified': False,
                'start_positions': []
            }
        
        # Start tracking for potential hold
        self.active_holds[slot] = {
            'start_time': time.time(),
            'start_x': 0,
            'start_y': 0,
            'notified': False
        }
        
        # Initialize motion stop tracking
        self.motion_stop_tracking[slot] = {
            'last_x': 0,
            'last_y': 0,
            'last_move_time': time.time()
        }
    
    def _handle_position_x(self, value: int):
        """Handle X coordinate changes."""
        slot = self.current_slot
        if slot in self.slot_data:
            self.slot_data[slot]['x'] = value
            
            if slot in self.fingers:
                sx, sy, _, _, st = self.fingers[slot]
                if sx == 0:
                    current_y = self.slot_data[slot].get('y', 0)
                    self.fingers[slot] = (value, current_y, value, current_y, st)
                else:
                    new_ex = value
                    new_ey = self.slot_data[slot]['y']
                    self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                    self.gesture_detector.track_motion(slot, new_ex, new_ey)
                    
                    # Store complete path data for Douglas-Peucker
                    if slot in self.finger_paths:
                        self.finger_paths[slot].append({
                            'x': float(new_ex),
                            'y': float(new_ey),
                            't': time.time(),
                            'slot': slot
                        })
                    
                    # Update motion stop tracking if significant movement
                    if slot in self.motion_stop_tracking:
                        last_x = self.motion_stop_tracking[slot]['last_x']
                        last_y = self.motion_stop_tracking[slot]['last_y']
                        dist = math.sqrt((new_ex - last_x)**2 + (new_ey - last_y)**2)
                        # print(f"DEBUG: Slot {slot} position update X, dist from last: {dist}px (threshold: {self.gesture_detector.DRAG_HOLD_STOP_MOVEMENT}px)")
                        if dist > self.gesture_detector.DRAG_HOLD_STOP_MOVEMENT:
                            # print(f"DEBUG: Significant movement detected for slot {slot}")
                            self.motion_stop_tracking[slot]['last_move_time'] = time.time()
                            self.motion_stop_tracking[slot]['last_x'] = new_ex
                            self.motion_stop_tracking[slot]['last_y'] = new_ey
            
            # Update hold tracking
            if slot in self.active_holds:
                if self.active_holds[slot]['start_x'] == 0:
                    self.active_holds[slot]['start_x'] = value
    
    def _handle_position_y(self, value: int):
        """Handle Y coordinate changes."""
        slot = self.current_slot
        if slot in self.slot_data:
            self.slot_data[slot]['y'] = value
            
            if slot in self.fingers:
                sx, sy, _, _, st = self.fingers[slot]
                if sy == 0:
                    current_x = self.slot_data[slot].get('x', 0)
                    self.fingers[slot] = (current_x, value, current_x, value, st)
                else:
                    new_ex = self.slot_data[slot]['x']
                    new_ey = value
                    self.fingers[slot] = (sx, sy, new_ex, new_ey, st)
                    self.gesture_detector.track_motion(slot, new_ex, new_ey)
                    
                    # Store complete path data for Douglas-Peucker
                    if slot in self.finger_paths:
                        self.finger_paths[slot].append({
                            'x': float(new_ex),
                            'y': float(new_ey),
                            't': time.time(),
                            'slot': slot
                        })
                    
                    # Update motion stop tracking if significant movement
                    if slot in self.motion_stop_tracking:
                        last_x = self.motion_stop_tracking[slot]['last_x']
                        last_y = self.motion_stop_tracking[slot]['last_y']
                        dist = math.sqrt((new_ex - last_x)**2 + (new_ey - last_y)**2)
                        # print(f"DEBUG: Slot {slot} position update Y, dist from last: {dist}px (threshold: {self.gesture_detector.DRAG_HOLD_STOP_MOVEMENT}px)")
                        if dist > self.gesture_detector.DRAG_HOLD_STOP_MOVEMENT:
                            # print(f"DEBUG: Significant movement detected for slot {slot}")
                            self.motion_stop_tracking[slot]['last_move_time'] = time.time()
                            self.motion_stop_tracking[slot]['last_x'] = new_ex
                            self.motion_stop_tracking[slot]['last_y'] = new_ey
            
            # Update hold tracking
            if slot in self.active_holds:
                if self.active_holds[slot]['start_y'] == 0:
                    self.active_holds[slot]['start_y'] = value
    
    def _check_gesture_state(self):
        """Check for hold and drag states."""
        current_time = time.time()
        
        # Check for hold
        if self.gesture_hold_state['start_time'] > 0 and not self.gesture_hold_state['notified']:
            self._check_hold_state(current_time)
        
        # Check for drag and hold
        if not self.gesture_hold_state['is_hold'] and not self.gesture_drag_hold_state['notified']:
            self._check_drag_hold_state(current_time)
        
        # Update pinch tracking only if we have actual finger movement
        if self.gesture_detector and self.fingers:
            # Only activate pinch tracking after fingers have moved
            has_moved = any(
                abs(ex - sx) > 5 or abs(ey - sy) > 5
                for sx, sy, ex, ey, _ in self.fingers.values()
            )
            if has_moved or self.gesture_detector.pinch_data['is_active']:
                self.gesture_detector.update_pinch_tracking(self.fingers)
    
    def _check_hold_state(self, current_time: float):
        """Check if a hold gesture should be triggered."""
        hold_duration = (current_time - self.gesture_hold_state['start_time']) * 1000
        
        effective_hold_timeout = self.gesture_detector.config.TAP_HOLD_TIMEOUT
        if len(self.active_slots) >= 2:
            effective_hold_timeout *= 0.8  # 20% shorter for multi-finger
        
        if hold_duration >= effective_hold_timeout:
            all_stable = True
            for slot in self.active_slots:
                if slot in self.fingers:
                    sx, sy, ex, ey, _ = self.fingers[slot]
                    movement = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                    if movement > self.gesture_detector.TAP_HOLD_MAX_MOVEMENT:
                        all_stable = False
                        break
            
            if all_stable:
                self.gesture_hold_state['is_hold'] = True
                self.gesture_hold_state['notified'] = True
                self._handle_hold_start()
    
    def _check_drag_hold_state(self, current_time: float):
        """Check if a drag and hold gesture should be triggered."""
        if not self.active_slots:
            return
        
        all_stable = True
        max_movement = 0
        has_initial_drag = self.gesture_drag_hold_state['initial_drag_detected']
        
        # Check for initial drag
        if not has_initial_drag:
            for slot in self.active_slots:
                if slot in self.fingers:
                    sx, sy, ex, ey, _ = self.fingers[slot]
                    movement = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                    max_movement = max(max_movement, movement)
                    # print(f"DEBUG: Slot {slot} initial movement: {movement}px (threshold: {self.gesture_detector.DRAG_HOLD_MIN_DRAG_DISTANCE}px)")
            
            if max_movement > self.gesture_detector.DRAG_HOLD_MIN_DRAG_DISTANCE:
                # print("DEBUG: Initial drag detected")
                self.gesture_drag_hold_state['initial_drag_detected'] = True
                self.gesture_drag_hold_state['drag_start_time'] = min(
                    st for _, _, _, _, st in self.fingers.values()
                )
                self.gesture_drag_hold_state['start_positions'] = [
                    (sx, sy) for sx, sy, _, _, _ in self.fingers.values()
                ]
        
        # Check for hold after drag
        if has_initial_drag and not self.gesture_drag_hold_state['notified']:
            stop_time = float('inf')
            for slot in self.active_slots:
                if slot in self.motion_stop_tracking:
                    time_since_move = current_time - self.motion_stop_tracking[slot]['last_move_time']
                    # print(f"DEBUG: Slot {slot} time since last move: {time_since_move * 1000:.0f}ms (threshold: {self.gesture_detector.config.DRAG_HOLD_TIMEOUT}ms)")
                    stop_time = min(stop_time, time_since_move)
                    if time_since_move < self.gesture_detector.config.DRAG_HOLD_TIMEOUT / 1000.0:
                        all_stable = False
                        # print(f"DEBUG: Slot {slot} not stable yet")
                        break
            
            # print(f"DEBUG: All stable: {all_stable}, Min stop time: {stop_time * 1000:.0f}ms")
            if all_stable and stop_time >= self.gesture_detector.config.DRAG_HOLD_TIMEOUT / 1000.0:
                # print("DEBUG: Drag hold confirmed")
                self.gesture_drag_hold_state['is_drag_hold'] = True
                self.gesture_drag_hold_state['notified'] = True
                self.gesture_drag_hold_state['stop_start_time'] = current_time - stop_time
                self.gesture_drag_hold_state['drag_end_positions'] = [
                    (ex, ey) for _, _, ex, ey, _ in self.fingers.values()
                ]
                
                # Log confirmation immediately when hold is confirmed
                self._handle_drag_hold_start()
                # Immediately classify and process the gesture
                gesture = self.gesture_detector.classify_gesture(
                    self.fingers,
                    is_hold=False,
                    is_drag_hold=True
                )
                self.logger.log_gesture(gesture)
    
    def _process_gesture(self):
        """Process the current gesture."""
        if not self.fingers:
            return
        
        # Skip processing if this is a drag and hold that was already confirmed
        if self.gesture_drag_hold_state['is_drag_hold'] and self.gesture_drag_hold_state['notified']:
            return
        
        # Skip processing if this is a hold that was already confirmed
        if self.gesture_hold_state['is_hold'] and self.gesture_hold_state['notified']:
            return
        
        gesture = self.gesture_detector.classify_gesture(
            self.fingers,
            is_hold=self.gesture_hold_state['is_hold'],
            is_drag_hold=self.gesture_drag_hold_state['is_drag_hold']
        )
        
        # Only log the gesture if it's not a hold or drag_hold (these are logged separately)
        if gesture.get('type') not in ['hold', 'drag_hold']:
            self.logger.log_gesture(gesture)
        
        # Clean up motion history
        self.gesture_detector.cleanup_motion_history()
    
    def _handle_hold_start(self):
        """Handle tap and hold start."""
        finger_count = len(self.fingers)
        positions = [(fx, fy) for _, _, fx, fy, _ in self.fingers.values()]
        
        self.logger.log_hold_start(finger_count, positions, self.gesture_hold_state['start_time'])
    
    def _handle_drag_hold_start(self):
        """Handle drag and hold start."""
        finger_count = len(self.fingers)
        positions = self.gesture_drag_hold_state['drag_end_positions']
        
        self.logger.log_drag_hold_start(
            finger_count, 
            positions, 
            self.gesture_drag_hold_state['start_positions'],
            self.gesture_drag_hold_state['drag_start_time']
        )
    
    def _hold_checker(self):
        """Periodic checker for hold states."""
        while self.running:
            time.sleep(0.01)  # Increased frequency for better responsiveness
            with self.state_lock:
                if self.active_slots:
                    self._check_gesture_state()

    def _reset_gesture_state(self):
        """Reset gesture state after processing."""
        self.fingers.clear()
        self.active_slots.clear()
        
        # Clean up all path storage
        self.finger_paths.clear()
        
        # Reset hold states but keep the timing info for next gesture
        self.gesture_hold_state = {
            'is_hold': False,
            'start_time': 0,
            'notified': False,
            'start_positions': []
        }
        
        self.gesture_drag_hold_state = {
            'is_drag_hold': False,
            'drag_start_time': 0,
            'stop_start_time': 0,
            'notified': False,
            'initial_drag_detected': False,
            'start_positions': [],
            'drag_end_positions': []
        }
        
        self.gesture_detector.pinch_data = {
            'initial_distance': 0.0,
            'min_distance': float('inf'),
            'max_distance': 0.0,
            'is_active': False
        }