#!/usr/bin/env python3
"""Test script for $Q Super-Quick Recognizer."""

import sys
import time
sys.path.insert(0, '.')

from touchscreen_listener.gestures.qdollar_recognizer import QDollarRecognizer


def test_gestures():
    """Test the $Q recognizer with various gestures."""
    recognizer = QDollarRecognizer()
    
    print("Testing $Q Super-Quick Recognizer")
    print("=" * 50)
    print(f"Loaded {len(recognizer.PointClouds)} point clouds")
    print()
    
    # Test gestures
    test_cases = [
        # T gesture
        ("T", [
            {'x': 100, 'y': 50, 't': 1},
            {'x': 200, 'y': 50, 't': 2},
            {'x': 150, 'y': 50, 't': 3},
            {'x': 150, 'y': 150, 't': 4}
        ]),
        
        # Line gesture
        ("line", [
            {'x': 50, 'y': 100, 't': 1},
            {'x': 200, 'y': 100, 't': 2}
        ]),
        
        # X gesture
        ("X", [
            {'x': 50, 'y': 50, 't': 1},
            {'x': 150, 'y': 150, 't': 2},
            {'x': 150, 'y': 50, 't': 3},
            {'x': 50, 'y': 150, 't': 4}
        ]),
        
        # H gesture
        ("H", [
            {'x': 50, 'y': 50, 't': 1},
            {'x': 50, 'y': 150, 't': 2},
            {'x': 50, 'y': 100, 't': 3},
            {'x': 150, 'y': 100, 't': 4},
            {'x': 150, 'y': 50, 't': 5},
            {'x': 150, 'y': 150, 't': 6}
        ]),
        
        # I gesture
        ("I", [
            {'x': 100, 'y': 50, 't': 1},
            {'x': 100, 'y': 150, 't': 2},
            {'x': 70, 'y': 50, 't': 3},
            {'x': 130, 'y': 50, 't': 4},
            {'x': 70, 'y': 150, 't': 5},
            {'x': 130, 'y': 150, 't': 6}
        ])
    ]
    
    # Run tests
    total_time = 0
    for expected_name, points in test_cases:
        print(f"Testing '{expected_name}' gesture...")
        
        # Test multiple times for average performance
        times = []
        results = []
        
        for _ in range(5):
            start_time = time.time()
            result = recognizer.recognize(points)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            results.append(result)
        
        avg_time = sum(times) / len(times)
        total_time += avg_time
        
        # Get the most common result
        result_names = [r.name for r in results]
        most_common = max(set(result_names), key=result_names.count)
        avg_score = sum(r.score for r in results if r.name == most_common) / result_names.count(most_common)
        
        print(f"  Expected: {expected_name}")
        print(f"  Recognized: {most_common}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Match: {'✓' if expected_name == most_common else '✗'}")
        print()
    
    print(f"Total average time per gesture: {total_time / len(test_cases):.2f}ms")
    
    # Test adding a custom gesture
    print("\nTesting custom gesture addition...")
    custom_points = [
        {'x': 100, 'y': 100, 't': 1},
        {'x': 120, 'y': 80, 't': 2},
        {'x': 140, 'y': 100, 't': 3},
        {'x': 120, 'y': 120, 't': 4},
        {'x': 100, 'y': 100, 't': 5}
    ]
    
    count = recognizer.add_gesture("diamond", custom_points)
    print(f"Added 'diamond' gesture. Total count: {count}")
    
    # Test recognizing the custom gesture
    result = recognizer.recognize(custom_points)
    print(f"Recognized custom gesture: {result.name} with score: {result.score:.3f}")
    
    # Test deleting user gestures
    deleted_count = recognizer.delete_user_gestures()
    print(f"Deleted user gestures. Remaining: {deleted_count}")
    
    print("\n$Q recognizer test completed!")


if __name__ == "__main__":
    test_gestures()