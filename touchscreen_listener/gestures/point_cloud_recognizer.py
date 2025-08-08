"""
$P (Point Cloud) Recognizer Implementation
A more advanced gesture recognizer that uses point cloud matching instead of $1's resampling approach.
"""

import math
import json
import os
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PointCloudRecognizer:
    """
    $P (Point Cloud) Recognizer implementation.
    
    This recognizer treats gestures as unordered point clouds and uses
    Euclidean distance matching with Golden Section Search for optimal alignment.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the Point Cloud Recognizer.
        
        Args:
            templates_path: Path to templates JSON file. If None, uses default.
        """
        self.templates = {}
        self.templates_path = templates_path or os.path.join(
            os.path.dirname(__file__), 'pc_templates.json'
        )
        self.load_templates()
    
    def load_templates(self):
        """Load gesture templates from JSON file."""
        try:
            with open(self.templates_path, 'r') as f:
                template_data = json.load(f)
                self.templates = {}
                for name, paths in template_data.items():
                    self.templates[name] = []
                    for path in paths:
                        points = [(p['x'], p['y']) for p in path]
                        points = self._normalize_points(points)
                        self.templates[name].append(points)
                logger.info(f"Loaded {len(self.templates)} gesture templates")
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
    
    def recognize(self, points: List[Tuple[float, float]], 
                  min_points: int = 3) -> Dict[str, any]:
        """
        Recognize a gesture from input points.
        
        Args:
            points: List of (x, y) coordinate tuples
            min_points: Minimum number of points required
            
        Returns:
            Dictionary with 'name', 'score', and 'distance' keys
        """
        if len(points) < min_points:
            return {'name': None, 'score': 0.0, 'distance': float('inf')}
        
        # Remove duplicate points and normalize
        points = self._remove_duplicates(points)
        points = self._normalize_points(points)
        
        best_score = 0.0
        best_gesture = None
        best_distance = float('inf')
        
        for gesture_name, template_list in self.templates.items():
            for template_points in template_list:
                distance = self._optimal_cloud_distance(points, template_points)
                score = 1 - (distance / 0.5)  # Convert distance to similarity score
                score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                
                if score > best_score:
                    best_score = score
                    best_gesture = gesture_name
                    best_distance = distance
        
        return {
            'name': best_gesture,
            'score': best_score,
            'distance': best_distance
        }
    
    def _remove_duplicates(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove consecutive duplicate points."""
        if not points:
            return []
        
        unique_points = [points[0]]
        for point in points[1:]:
            if point != unique_points[-1]:
                unique_points.append(point)
        return unique_points
    
    def _normalize_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Normalize points to a standard scale and position.
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            Normalized points centered at origin with unit scale
        """
        if not points:
            return []
        
        # Calculate centroid
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)
        
        # Translate to origin
        translated = [(x - centroid_x, y - centroid_y) for x, y in points]
        
        # Scale to unit size
        max_distance = max(math.sqrt(x*x + y*y) for x, y in translated)
        if max_distance > 0:
            scale = 1.0 / max_distance
            translated = [(x * scale, y * scale) for x, y in translated]
        
        return translated
    
    def _cloud_distance(self, points1: List[Tuple[float, float]], 
                       points2: List[Tuple[float, float]]) -> float:
        """
        Calculate the distance between two point clouds using the $P algorithm.
        
        Args:
            points1: First point cloud
            points2: Second point cloud
            
        Returns:
            Distance between the point clouds
        """
        if not points1 or not points2:
            return float('inf')
        
        d1 = self._calculate_distance(points1, points2)
        d2 = self._calculate_distance(points2, points1)
        
        return (d1 + d2) / 2.0    

    def _calculate_distance(self, points1: List[Tuple[float, float]], 
                          points2: List[Tuple[float, float]]) -> float:
        """
        Calculate directed distance from points1 to points2.
        
        Args:
            points1: Source points
            points2: Target points
            
        Returns:
            Directed distance
        """
        if not points1 or not points2:
            return float('inf')
        
        total_distance = 0.0
        
        for p1 in points1:
            min_distance = float('inf')
            for p2 in points2:
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        
        return total_distance / len(points1)
    
    def _rotate_points(self, points: List[Tuple[float, float]], theta: float) -> List[Tuple[float, float]]:
        """Rotate points around origin by theta radians."""
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return [(x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta) for x, y in points]

    def _golden_section_search(self, f, a: float, b: float, tol: float = 0.01) -> float:
        """Golden Section Search to find minimum of f in [a, b]."""
        gr = (1 + math.sqrt(5)) / 2
        while abs(b - a) > tol:
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            if f(c) < f(d):
                b = d
            else:
                a = c
        return (a + b) / 2

    def _optimal_cloud_distance(self, points: List[Tuple[float, float]], 
                              template_points: List[Tuple[float, float]]) -> float:
        """Compute minimal cloud distance using GSS for rotation."""
        def distance_at_theta(theta):
            rotated = self._rotate_points(points, theta)
            return self._cloud_distance(rotated, template_points)
        
        a = -math.pi / 4  # -45 degrees
        b = math.pi / 4   # 45 degrees
        best_theta = self._golden_section_search(distance_at_theta, a, b)
        return distance_at_theta(best_theta)

    def add_template(self, name: str, points: List[Tuple[float, float]]):
        """
        Add a new gesture template.
        
        Args:
            name: Gesture name
            points: List of (x, y) coordinate tuples
        """
        if name not in self.templates:
            self.templates[name] = []
        
        normalized_points = self._normalize_points(points)
        self.templates[name].append(normalized_points)
        logger.info(f"Added template for gesture '{name}'")
    
    def save_templates(self):
        """Save current templates to JSON file."""
        template_data = {}
        for name, template_list in self.templates.items():
            template_data[name] = []
            for points in template_list:
                path = [{'x': x, 'y': y} for x, y in points]
                template_data[name].append(path)
        
        try:
            with open(self.templates_path, 'w') as f:
                json.dump(template_data, f, indent=2)
            logger.info("Templates saved successfully")
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")


class PointCloudGestureDetector:
    """
    High-level gesture detector that uses the Point Cloud Recognizer.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the gesture detector.
        
        Args:
            threshold: Minimum similarity score for recognition
        """
        self.recognizer = PointCloudRecognizer()
        self.threshold = threshold
    
    def detect_gesture(self, path: List[Tuple[float, float]]) -> Dict[str, any]:
        """
        Detect a gesture from a path.
        
        Args:
            path: List of (x, y) coordinate tuples
            
        Returns:
            Dictionary with gesture information
        """
        result = self.recognizer.recognize(path)
        
        if result['score'] >= self.threshold:
            return {
                'gesture': result['name'],
                'confidence': result['score'],
                'recognized': True
            }
        else:
            return {
                'gesture': None,
                'confidence': result['score'],
                'recognized': False
            }
    
    def set_threshold(self, threshold: float):
        """Set the recognition threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def add_template(self, name: str, path: List[Tuple[float, float]]):
        """Add a new gesture template."""
        self.recognizer.add_template(name, path)
    
    def save_templates(self):
        """Save templates to file."""
        self.recognizer.save_templates()