"""
Shared utilities for gesture recognition and processing.

This module provides common functionality used across different classifiers
and gesture processing components to eliminate code duplication.
"""

import math
import time
from typing import List, Dict, Tuple, Optional


class Point:
    """Represents a 2D point with optional timestamp."""
    
    def __init__(self, x: float, y: float, timestamp: Optional[float] = None):
        self.x = float(x)
        self.y = float(y)
        self.t = timestamp or time.time()
        self.timestamp = self.t  # Alias for compatibility
    
    def __repr__(self):
        return f"Point({self.x:.1f}, {self.y:.1f})"
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class GeometryUtils:
    """Utility class for geometric calculations."""
    
    @staticmethod
    def calculate_centroid(points: List[Point]) -> Point:
        """Calculate the centroid of a list of points."""
        if not points:
            return Point(0, 0)
        sum_x = sum(p.x for p in points)
        sum_y = sum(p.y for p in points)
        return Point(sum_x / len(points), sum_y / len(points))
    
    @staticmethod
    def rotate_points(points: List[Point], angle: float, centroid: Optional[Point] = None) -> List[Point]:
        """Rotate points around a centroid."""
        if centroid is None:
            centroid = GeometryUtils.calculate_centroid(points)
        
        rotated = []
        for point in points:
            dx = point.x - centroid.x
            dy = point.y - centroid.y
            
            new_x = dx * math.cos(angle) - dy * math.sin(angle) + centroid.x
            new_y = dx * math.sin(angle) + dy * math.cos(angle) + centroid.y
            
            rotated.append(Point(new_x, new_y))
        
        return rotated
    
    @staticmethod
    def calculate_distance(p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    @staticmethod
    def calculate_path_length(points: List[Point]) -> float:
        """Calculate total path length."""
        if len(points) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(points)):
            length += GeometryUtils.calculate_distance(points[i-1], points[i])
        return length
    
    @staticmethod
    def calculate_perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate perpendicular distance from point to line segment."""
        # Handle case where start and end are the same point
        if abs(line_start.x - line_end.x) < 1e-10 and abs(line_start.y - line_end.y) < 1e-10:
            return GeometryUtils.calculate_distance(point, line_start)
        
        # Calculate line equation: Ax + By + C = 0
        A = line_end.y - line_start.y
        B = line_start.x - line_end.x
        C = line_end.x * line_start.y - line_start.x * line_end.y
        
        denominator = math.sqrt(A * A + B * B)
        if denominator < 1e-10:
            return GeometryUtils.calculate_distance(point, line_start)
        
        return abs(A * point.x + B * point.y + C) / denominator


class VelocityCalculator:
    """Utility class for velocity calculations."""
    
    @staticmethod
    def calculate_velocity(p1: Point, p2: Point, dt: float) -> float:
        """Calculate velocity between two points."""
        if dt <= 0:
            return 0.0
        distance = GeometryUtils.calculate_distance(p1, p2)
        return distance / dt
    
    @staticmethod
    def calculate_average_velocity(path: List[Dict[str, float]]) -> float:
        """Calculate average velocity of a path."""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(path)):
            dx = path[i]['x'] - path[i-1]['x']
            dy = path[i]['y'] - path[i-1]['y']
            dt = max(path[i]['t'] - path[i-1]['t'], 1.0)  # Prevent division by zero
            
            total_distance += math.sqrt(dx * dx + dy * dy)
            total_time += dt
        
        return total_distance / total_time if total_time > 0 else 0.0
    
    @staticmethod
    def calculate_velocity_statistics(path: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate comprehensive velocity statistics."""
        if len(path) < 2:
            return {
                'avg_velocity': 0.0,
                'max_velocity': 0.0,
                'velocity_variance': 0.0,
                'min_velocity': 0.0
            }
        
        velocities = []
        for i in range(1, len(path)):
            dx = path[i]['x'] - path[i-1]['x']
            dy = path[i]['y'] - path[i-1]['y']
            dt = max(path[i]['t'] - path[i-1]['t'], 1.0)
            
            if dt > 0:
                velocity = math.sqrt(dx * dx + dy * dy) / dt
                velocities.append(velocity)
        
        if not velocities:
            return {
                'avg_velocity': 0.0,
                'max_velocity': 0.0,
                'velocity_variance': 0.0,
                'min_velocity': 0.0
            }
        
        avg_velocity = sum(velocities) / len(velocities)
        max_velocity = max(velocities)
        min_velocity = min(velocities)
        
        # Calculate variance
        if len(velocities) > 1:
            variance = sum((v - avg_velocity) ** 2 for v in velocities) / (len(velocities) - 1)
        else:
            variance = 0.0
        
        return {
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'velocity_variance': variance,
            'min_velocity': min_velocity
        }


class PathUtils:
    """Utility class for path processing."""
    
    @staticmethod
    def convert_dict_to_points(path: List[Dict[str, float]]) -> List[Point]:
        """Convert path from dict format to Point objects."""
        return [Point(p['x'], p['y'], p.get('t', 0)) for p in path]
    
    @staticmethod
    def convert_points_to_dict(points: List[Point]) -> List[Dict[str, float]]:
        """Convert Point objects to dict format."""
        return [{'x': p.x, 'y': p.y, 't': p.t} for p in points]
    
    @staticmethod
    def get_path_bounds(points: List[Point]) -> Tuple[float, float, float, float]:
        """Get bounding box of a path."""
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        return min_x, max_x, min_y, max_y
    
    @staticmethod
    def calculate_path_duration(path: List[Dict[str, float]]) -> float:
        """Calculate total duration of a path."""
        if len(path) < 2:
            return 0.0
        return path[-1]['t'] - path[0]['t']


class DataValidator:
    """Utility class for validating gesture data."""
    
    @staticmethod
    def validate_path_data(path: List[Dict[str, float]]) -> bool:
        """Validate that path data has required structure."""
        if not isinstance(path, list) or len(path) < 2:
            return False
        
        for point in path:
            if not isinstance(point, dict):
                return False
            if 'x' not in point or 'y' not in point:
                return False
            try:
                float(point['x'])
                float(point['y'])
                if 't' in point:
                    float(point['t'])
            except (ValueError, TypeError):
                return False
        
        return True
    
    @staticmethod
    def sanitize_path_data(path: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Sanitize path data by ensuring all required fields exist."""
        sanitized = []
        for point in path:
            sanitized_point = {
                'x': float(point.get('x', 0)),
                'y': float(point.get('y', 0)),
                't': float(point.get('t', 0))
            }
            sanitized.append(sanitized_point)
        return sanitized