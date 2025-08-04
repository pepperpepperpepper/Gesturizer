#!/usr/bin/env python3
"""
Touchscreen Listener - Main Entry Point
A modular touchscreen event listener for gesture recognition.
"""

import time
from touchscreen_listener.core.listener import TouchListener

def main():
    """Main entry point for the touchscreen listener."""
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