"""
Statistical Classifier for Gesture vs Drawing Classification

Uses statistical analysis and adaptive thresholds based on actual data patterns.
No external dependencies - pure Python implementation.
"""

import math
import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from ..utils.gesture_utils import VelocityCalculator


class StatisticalClassifier:
    """
    Statistical classifier that learns from data patterns.
    
    Uses multiple features with adaptive thresholds based on statistical
    properties rather than hard-coded values.
    """
    
    def __init__(self, data_file: str = "gesture_stats.json"):
        self.data_file = data_file
        self.gesture_stats = defaultdict(list)
        self.drawing_stats = defaultdict(list)
        self.is_trained = False
        self._load_training_data()
    
    def _extract_features(self, path: List[Dict[str, float]]) -> Dict[str, float]:
        """Extract statistical features from the path."""
        if len(path) < 2:
            return {
                'duration': 0,
                'avg_velocity': 0,
                'max_velocity': 0,
                'velocity_variance': 0,
                'total_distance': 0,
                'straight_line_distance': 0,
                'direction_changes': 0,
                'compactness': 0,
                'point_density': 0,
                'curvature': 0,
                'start_end_distance': 0
            }
        
        # Basic measurements
        num_points = len(path)
        duration = path[-1]['t'] - path[0]['t']
        
        # Use shared velocity calculator
        velocity_stats = VelocityCalculator.calculate_velocity_statistics(path)
        avg_velocity = velocity_stats['avg_velocity']
        max_velocity = velocity_stats['max_velocity']
        velocity_variance = velocity_stats['velocity_variance']
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(path)):
            dx = path[i]['x'] - path[i-1]['x']
            dy = path[i]['y'] - path[i-1]['y']
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        # Direction changes
        direction_changes = 0
        if len(path) > 2:
            prev_dx = path[1]['x'] - path[0]['x']
            prev_dy = path[1]['y'] - path[0]['y']
            
            for i in range(2, len(path)):
                dx = path[i]['x'] - path[i-1]['x']
                dy = path[i]['y'] - path[i-1]['y']
                
                # Calculate angle change
                dot_product = prev_dx * dx + prev_dy * dy
                mag1 = math.sqrt(prev_dx**2 + prev_dy**2)
                mag2 = math.sqrt(dx**2 + dy**2)
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                    angle = math.acos(cos_angle)
                    
                    if angle > math.pi / 3:  # Significant direction change
                        direction_changes += 1
                
                prev_dx, prev_dy = dx, dy
        
        # Calculate bounding box
        x_coords = [p['x'] for p in path]
        y_coords = [p['y'] for p in path]
        width = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
        height = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
        
        # Compactness (path length vs bounding box diagonal)
        diagonal = math.sqrt(width**2 + height**2) if width > 0 and height > 0 else 1
        compactness = total_distance / diagonal if diagonal > 0 else 1
        
        # Point density
        point_density = num_points / max(duration, 1.0) if duration > 0 else 0
        
        return {
            'num_points': num_points,
            'duration': duration,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'velocity_variance': velocity_variance,
            'direction_changes': direction_changes,
            'compactness': compactness,
            'point_density': point_density,
            'path_length': total_distance,
            'width': width,
            'height': height
        }
    
    def _variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    
    def _load_training_data(self):
        """Load training data from file or use defaults."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.gesture_stats = defaultdict(list, data.get('gesture', {}))
                    self.drawing_stats = defaultdict(list, data.get('drawing', {}))
                    self.is_trained = len(self.gesture_stats) > 0 and len(self.drawing_stats) > 0
            except Exception:
                pass
        
        if not self.is_trained:
            self._generate_default_training_data()
    
    def _generate_default_training_data(self):
        """Generate default training data based on typical patterns."""
        # Gesture patterns (quick, simple movements)
        gesture_examples = [
            # Quick swipe right
            [{'x': 100, 'y': 100, 't': 0}, {'x': 200, 'y': 100, 't': 0.1}],
            # Quick swipe down
            [{'x': 100, 'y': 100, 't': 0}, {'x': 100, 'y': 200, 't': 0.15}],
            # Quick diagonal
            [{'x': 50, 'y': 50, 't': 0}, {'x': 150, 'y': 150, 't': 0.12}],
            # Quick tap-like
            [{'x': 100, 'y': 100, 't': 0}, {'x': 100, 'y': 100, 't': 0.05}],
        ]
        
        # Drawing patterns (slow, complex movements)
        drawing_examples = [
            # Slow circle
            [{'x': 100 + 30*math.cos(i*0.2), 'y': 100 + 30*math.sin(i*0.2), 't': i*0.1} 
             for i in range(32)],
            # Complex shape
            [{'x': 100 + 20*math.cos(i*0.3), 'y': 100 + 20*math.sin(i*0.4), 't': i*0.08} 
             for i in range(25)],
            # Slow line
            [{'x': 50, 'y': 50, 't': 0}] + [{'x': 50 + i*2, 'y': 50, 't': i*0.05} 
                                           for i in range(50)],
        ]
        
        # Add examples to training data
        for path in gesture_examples:
            features = self._extract_features(path)
            for key, value in features.items():
                self.gesture_stats[key].append(value)
        
        for path in drawing_examples:
            features = self._extract_features(path)
            for key, value in features.items():
                self.drawing_stats[key].append(value)
        
        self.is_trained = True
        self._save_training_data()
    
    def _save_training_data(self):
        """Save training data to file."""
        data = {
            'gesture': dict(self.gesture_stats),
            'drawing': dict(self.drawing_stats)
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate mean and std for a list of values."""
        if not values:
            return {'mean': 0, 'std': 1}
        
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            std = 1
        
        return {'mean': mean, 'std': max(std, 0.001)}  # Avoid zero std
    
    def _calculate_probability(self, value: float, gesture_values: List[float], 
                               drawing_values: List[float]) -> float:
        """Calculate probability that value belongs to gesture class."""
        if not gesture_values or not drawing_values:
            return 0.5
        
        # Simple Gaussian probability calculation
        gesture_stats = self._calculate_statistics(gesture_values)
        drawing_stats = self._calculate_statistics(drawing_values)
        
        # Gaussian probability density
        def gaussian_pdf(x, mean, std):
            return math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
        
        p_gesture = gaussian_pdf(value, gesture_stats['mean'], gesture_stats['std'])
        p_drawing = gaussian_pdf(value, drawing_stats['mean'], drawing_stats['std'])
        
        # Normalize to get probability
        total = p_gesture + p_drawing
        return p_gesture / total if total > 0 else 0.5
    
    def classify_path(self, path: List[Dict[str, float]]) -> str:
        """Classify a path as 'gesture' or 'drawing'."""
        if not self.is_trained:
            return 'gesture'  # Fallback
        
        features = self._extract_features(path)
        if not features:
            return 'gesture'
        
        # Calculate weighted probability across all features
        probabilities = []
        weights = {
            'duration': 0.3,
            'avg_velocity': 0.25,
            'direction_changes': 0.2,
            'compactness': 0.15,
            'point_density': 0.1
        }
        
        for feature_name, weight in weights.items():
            if feature_name in features and feature_name in self.gesture_stats and feature_name in self.drawing_stats:
                prob = self._calculate_probability(
                    features[feature_name],
                    self.gesture_stats[feature_name],
                    self.drawing_stats[feature_name]
                )
                probabilities.append(prob * weight)
        
        if not probabilities:
            return 'gesture'
        
        avg_probability = sum(probabilities) / sum(weights.values())
        
        # Threshold for classification
        return 'gesture' if avg_probability > 0.5 else 'drawing'
    
    def add_training_example(self, path: List[Dict[str, float]], label: str):
        """Add a new training example."""
        features = self._extract_features(path)
        if not features:
            return
        
        target_stats = self.gesture_stats if label == 'gesture' else self.drawing_stats
        for key, value in features.items():
            target_stats[key].append(value)
        
        self.is_trained = True
        self._save_training_data()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance of different features based on training data."""
        importance = {}
        
        for feature in ['duration', 'avg_velocity', 'direction_changes', 'compactness', 'point_density']:
            if feature in self.gesture_stats and feature in self.drawing_stats:
                gesture_stats = self._calculate_statistics(self.gesture_stats[feature])
                drawing_stats = self._calculate_statistics(self.drawing_stats[feature])
                
                # Calculate separation between distributions
                separation = abs(gesture_stats['mean'] - drawing_stats['mean']) / \
                           max(gesture_stats['std'], drawing_stats['std'])
                importance[feature] = separation
        
        return importance


# Global instance
_classifier = None

def classify_path(path: List[Dict[str, float]]) -> str:
    """Global function to classify a path."""
    global _classifier
    if _classifier is None:
        _classifier = StatisticalClassifier()
    return _classifier.classify_path(path)


def add_training_example(path: List[Dict[str, float]], label: str):
    """Add a new training example."""
    global _classifier
    if _classifier is None:
        _classifier = StatisticalClassifier()
    _classifier.add_training_example(path, label)


def get_feature_importance() -> Dict[str, float]:
    """Get feature importance."""
    global _classifier
    if _classifier is None:
        _classifier = StatisticalClassifier()
    return _classifier.get_feature_importance()