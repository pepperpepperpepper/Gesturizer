"""
Douglas-Peucker path simplification algorithm implementation.

This module provides efficient path simplification for touch gesture data,
reducing the number of points while preserving the overall shape.
"""

import math
import time
from typing import List, Tuple, Optional


class Point:
    """Represents a 2D point with optional timestamp."""
    
    def __init__(self, x: float, y: float, timestamp: Optional[float] = None):
        self.x = x
        self.y = y
        self.t = timestamp or time.time()
        self.timestamp = self.t  # Alias for compatibility
    
    def __repr__(self):
        return f"Point({self.x:.1f}, {self.y:.1f})"


class DouglasPeucker:
    """
    Douglas-Peucker path simplification algorithm.
    
    This implementation is optimized for touch gesture data, providing
    fast simplification while preserving important gesture features.
    """
    
    def __init__(self, epsilon: float = 15.0):
        """
        Initialize the Douglas-Peucker algorithm.
        
        Args:
            epsilon: Maximum distance threshold (pixels) for point elimination.
                    Higher values = more aggressive simplification.
        """
        self.epsilon = max(1.0, epsilon)  # Ensure minimum epsilon of 1px
    
    def simplify(self, points: List[Point]) -> List[Point]:
        """
        Simplify a path using the Douglas-Peucker algorithm.
        
        Args:
            points: List of Point objects representing the path
            
        Returns:
            List of simplified Point objects
            
        Performance:
            O(n log n) average case, O(n²) worst case
            Target: <5ms for 100-point paths
        """
        if not points:
            return []
        
        if len(points) <= 2:
            return points.copy()
        
        # Convert to list of tuples for processing
        coords = [(p.x, p.y) for p in points]
        
        # Run Douglas-Peucker
        simplified_coords = self._douglas_peucker(coords, self.epsilon)
        
        # Convert back to Points, preserving original timestamps where possible
        result = []
        for x, y in simplified_coords:
            # Find closest original point to preserve timestamp
            closest_point = min(points, key=lambda p: abs(p.x - x) + abs(p.y - y))
            timestamp = getattr(closest_point, 'timestamp', None) or getattr(closest_point, 't', time.time())
            result.append(Point(x, y, timestamp))
        
        return result
    
    def _douglas_peucker(self, points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
        """
        Core Douglas-Peucker algorithm implementation.
        
        Args:
            points: List of (x, y) coordinate tuples
            epsilon: Distance threshold
            
        Returns:
            Simplified list of coordinate tuples
        """
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line between start and end
        max_dist = 0.0
        max_index = 0
        
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        
        # Handle case where start and end are the same point
        if abs(start_x - end_x) < 1e-10 and abs(start_y - end_y) < 1e-10:
            return [points[0], points[-1]]
        
        # Calculate line equation: Ax + By + C = 0
        # Line from (start_x, start_y) to (end_x, end_y)
        A = end_y - start_y
        B = start_x - end_x
        C = end_x * start_y - start_x * end_y
        
        # Handle vertical or horizontal lines
        denominator = math.sqrt(A * A + B * B)
        if denominator < 1e-10:
            return [points[0], points[-1]]
        
        # Find point with maximum perpendicular distance
        for i in range(1, len(points) - 1):
            x, y = points[i]
            
            # Perpendicular distance from point to line
            dist = abs(A * x + B * y + C) / denominator
            
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursive call for the two segments
            left_points = self._douglas_peucker(points[:max_index + 1], epsilon)
            right_points = self._douglas_peucker(points[max_index:], epsilon)
            
            # Combine results (avoiding duplicate point at max_index)
            return left_points[:-1] + right_points
        else:
            # All intermediate points are within epsilon, return just endpoints
            return [points[0], points[-1]]
    
    def get_compression_ratio(self, original: List[Point], simplified: List[Point]) -> float:
        """
        Calculate the compression ratio achieved by simplification.
        
        Args:
            original: Original list of points
            simplified: Simplified list of points
            
        Returns:
            Compression ratio (original_length / simplified_length)
        """
        if not original or not simplified:
            return 1.0
        return len(original) / len(simplified)
    
    def benchmark(self, points: List[Point], iterations: int = 100) -> dict:
        """
        Benchmark the simplification performance.
        
        Args:
            points: Points to benchmark with
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing and compression statistics
        """
        if not points:
            return {"error": "No points provided"}
        
        times = []
        compression_ratios = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            simplified = self.simplify(points)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            compression_ratios.append(self.get_compression_ratio(points, simplified))
        
        return {
            "original_points": len(points),
            "simplified_points": len(simplified),
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios),
            "epsilon": self.epsilon
        }


def douglas_peucker(points: List[Point], epsilon: float = 15.0) -> List[Point]:
    """
    Convenience function for Douglas-Peucker simplification.
    
    Args:
        points: List of Point objects to simplify
        epsilon: Distance threshold in pixels (default: 15.0)
        
    Returns:
        Simplified list of Point objects
        
    Example:
        >>> points = [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
        >>> simplified = douglas_peucker(points, epsilon=2.0)
        >>> len(simplified) <= len(points)
        True
    """
    dp = DouglasPeucker(epsilon)
    return dp.simplify(points)


# Configuration presets for different use cases
SIMPLIFICATION_PRESETS = {
    "minimal": {"epsilon": 5.0, "description": "Minimal simplification, preserve details"},
    "balanced": {"epsilon": 15.0, "description": "Balanced simplification (default)"},
    "aggressive": {"epsilon": 30.0, "description": "Aggressive simplification, reduce data"},
    "extreme": {"epsilon": 50.0, "description": "Extreme simplification, minimal points"}
}


def get_preset_config(preset_name: str) -> dict:
    """
    Get configuration for a named preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Dictionary with epsilon value and description
    """
    return SIMPLIFICATION_PRESETS.get(preset_name, SIMPLIFICATION_PRESETS["balanced"])