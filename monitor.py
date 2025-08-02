#!/usr/bin/env python3
"""
Real-time touchscreen path storage monitor.
Shows live path data as you draw on your touchscreen.
"""

import time
import sys
import os
from touchscreen_listener.core.listener import TouchListener

class PathMonitor:
    def __init__(self):
        self.listener = TouchListener()
        self.running = False
    
    def start(self):
        """Start monitoring touchscreen paths."""
        if not self.listener.start():
            print("❌ No touchscreen found")
            return False
        
        self.running = True
        print("🎯 Touchscreen Path Monitor Started")
        print("=" * 50)
        print("📱 Touch and drag on your screen to see path data")
        print("🖱️  Press Ctrl+C to stop")
        print()
        
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        self.listener.stop()
        print("\n✅ Monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        last_path_counts = {}
        
        while self.running:
            # Get current path data
            current_paths = self.listener.finger_paths
            
            if current_paths != last_path_counts:
                self._display_paths(current_paths)
                last_path_counts = {k: len(v) for k, v in current_paths.items()}
            
            time.sleep(0.1)  # Update every 100ms
    
    def _display_paths(self, paths):
        """Display current path information."""
        if not paths:
            print("🤏 No fingers touching...", end="\r")
            return
        
        # Clear line and show current paths
        print("\r" + " " * 80 + "\r", end="")
        
        for slot, path in paths.items():
            if path:
                points = len(path)
                duration = path[-1]['t'] - path[0]['t']
                
                # Calculate bounding box
                min_x = min(p['x'] for p in path)
                max_x = max(p['x'] for p in path)
                min_y = min(p['y'] for p in path)
                max_y = max(p['y'] for p in path)
                
                print(f"👆 Finger {slot}: {points:3d} pts | "
                      f"{duration:.2f}s | "
                      f"({min_x:4.0f},{min_y:4.0f})→({max_x:4.0f},{max_y:4.0f})")
            else:
                print(f"👆 Finger {slot}: 0 pts")

def main():
    """Main entry point."""
    monitor = PathMonitor()
    monitor.start()

if __name__ == "__main__":
    main()