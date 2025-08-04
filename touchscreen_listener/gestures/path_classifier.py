"""
Path Classification System for Touchscreen Gestures

This module provides functionality to classify touch paths as either
quick gestures or deliberate drawing based on path characteristics.
"""

import math
from typing import List, Dict, Literal, Tuple
from ..utils.gesture_utils import VelocityCalculator


class PathClassifier:
    """
    Classifies touch paths as either 'gesture' or 'drawing' based on
    various characteristics like point count, duration, velocity, and curvature.
    """
    
    def __init__(self):
        # Use hybrid classifier combining $1 Recognizer and Statistical ML
        from touchscreen_listener.gestures.hybrid_classifier import HybridClassifier
        self.classifier = HybridClassifier(use_templates=True, use_ml=True)
        
    def classify_path(self, path: List[Dict[str, float]]) -> Literal['gesture', 'drawing']:
        """
        Classify a path as either a quick gesture or deliberate drawing.
        
        Args:
            path: List of dicts with 'x', 'y', 't' keys representing path points
            
        Returns:
            'gesture' for quick gestures, 'drawing' for deliberate drawing
            
        Raises:
            ValueError: If path is empty or malformed
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        if len(path) < 2:
            return 'gesture'  # Single point or very short path
            
        # Validate path format
        self._validate_path(path)
        
        # Use ML classifier
        return self.classifier.classify_path(path)
    
    def _validate_path(self, path: List[Dict[str, float]]) -> None:
        """Validate that path has correct format."""
        required_keys = {'x', 'y', 't'}
        for point in path:
            if not isinstance(point, dict):
                raise ValueError("Each path point must be a dictionary")
            if not required_keys.issubset(point.keys()):
                raise ValueError(f"Each point must contain {required_keys}")
            for key in required_keys:
                if not isinstance(point[key], (int, float)):
                    raise ValueError(f"Point {key} must be numeric")
    
    def _advanced_classification(self, path: List[Dict[str, float]]) -> Literal['gesture', 'drawing']:
        """Advanced classification using velocity and curvature analysis."""
        velocities = self._calculate_velocities(path)
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        
        # High velocity indicates quick gesture
        if avg_velocity > self.velocity_threshold:
            return 'gesture'
            
        # Analyze curvature - smooth curves suggest drawing
        curvature = self._calculate_curvature(path)
        if curvature < self.curvature_threshold:
            return 'drawing'
            
        # Default to gesture for ambiguous cases
        return 'gesture'
    
    def _calculate_velocities(self, path: List[Dict[str, float]]) -> List[float]:
        """Calculate velocities between consecutive points."""
        velocities = []
        
        for i in range(1, len(path)):
            prev_point = path[i-1]
            curr_point = path[i]
            
            # Calculate distance
            dx = curr_point['x'] - prev_point['x']
            dy = curr_point['y'] - prev_point['y']
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calculate time delta
            dt = (curr_point['t'] - prev_point['t']) * 1000  # ms
            if dt <= 0:
                continue  # Skip invalid time deltas
                
            velocity = distance / dt
            velocities.append(velocity)
            
        return velocities
    
    def _calculate_curvature(self, path: List[Dict[str, float]]) -> float:
        """
        Calculate average curvature of the path.
        Returns normalized curvature value (0 = straight, 1 = very curved)
        """
        if len(path) < 3:
            return 0.0  # Straight line has no curvature
            
        total_curvature = 0.0
        curvature_count = 0
        
        # Calculate curvature at each interior point
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]
            
            # Calculate vectors
            v1x = curr_point['x'] - prev_point['x']
            v1y = curr_point['y'] - prev_point['y']
            v2x = next_point['x'] - curr_point['x']
            v2y = next_point['y'] - curr_point['y']
            
            # Calculate cross product magnitude (2D equivalent of cross product)
            cross_product = abs(v1x * v2y - v1y * v2x)
            
            # Calculate magnitudes
            mag1 = math.sqrt(v1x * v1x + v1y * v1y)
            mag2 = math.sqrt(v2x * v2x + v2y * v2y)
            
            if mag1 > 0 and mag2 > 0:
                # Normalize curvature by segment lengths
                curvature = cross_product / (mag1 * mag2)
                total_curvature += curvature
                curvature_count += 1
        
        if curvature_count == 0:
            return 0.0
            
        avg_curvature = total_curvature / curvature_count
        
        # Normalize to 0-1 range based on typical values
        return min(avg_curvature / 0.5, 1.0)
    
    def get_path_stats(self, path: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Get detailed statistics about a path for debugging/analysis.
        
        Returns:
            Dictionary with point_count, duration, avg_velocity, max_velocity,
            curvature, total_distance, bounding_box
        """
        if not path:
            return {}
            
        point_count = len(path)
        duration = (path[-1]['t'] - path[0]['t']) * 1000  # Convert to ms
        velocities = self._calculate_velocities(path)
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        max_velocity = max(velocities) if velocities else 0
        curvature = self._calculate_curvature(path)
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(path)):
            dx = path[i]['x'] - path[i-1]['x']
            dy = path[i]['y'] - path[i-1]['y']
            total_distance += math.sqrt(dx * dx + dy * dy)
        
        # Calculate bounding box
        min_x = min(p['x'] for p in path)
        max_x = max(p['x'] for p in path)
        min_y = min(p['y'] for p in path)
        max_y = max(p['y'] for p in path)
        
        return {
            'point_count': point_count,
            'duration': duration,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'curvature': curvature,
            'total_distance': total_distance,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'width': max_x - min_x,
            'height': max_y - min_y
        }


# Convenience function for simple usage
def classify_path(path: List[Dict[str, float]]) -> Literal['gesture', 'drawing']:
    """
    Simple interface to classify a path.
    
    Args:
        path: List of dicts with 'x', 'y', 't' keys
        
    Returns:
        'gesture' or 'drawing'
    """
    classifier = PathClassifier()
    return classifier.classify_path(path)


def get_path_stats(path: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Get detailed statistics about a path.
    
    Args:
        path: List of dicts with 'x', 'y', 't' keys
        
    Returns:
        Dictionary with path statistics
    """
    classifier = PathClassifier()
    return classifier.get_path_stats(path)