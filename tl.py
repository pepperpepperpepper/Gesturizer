#!/usr/bin/env python3
"""
Touchscreen Listener (tl.py) - Legacy compatibility wrapper
Maintains backward compatibility with the original single-file approach.
"""

from touchscreen_listener.core.listener import TouchListener
import time

def main():
    """Legacy main function for backward compatibility."""
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