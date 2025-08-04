"""
Hybrid Gesture Classifier

Combines $1 Unistroke Recognizer with Statistical Machine Learning
for robust gesture vs drawing classification.
"""

import math
from typing import List, Dict, Literal
from touchscreen_listener.gestures.dollar_recognizer import DollarRecognizer
from touchscreen_listener.gestures.statistical_classifier import StatisticalClassifier
from ..utils.gesture_utils import VelocityCalculator


class HybridClassifier:
    """
    Hybrid classifier that combines $1 Recognizer and Statistical ML.
    
    Uses $1 Recognizer for template matching and Statistical Classifier
    for learned patterns, with a confidence-based voting mechanism.
    """
    
    def __init__(self, use_templates: bool = True, use_ml: bool = True):
        """
        Initialize hybrid classifier.
        
        Args:
            use_templates: Whether to use $1 Recognizer templates
            use_ml: Whether to use statistical ML classifier
        """
        self.use_templates = use_templates
        self.use_ml = use_ml
        
        if use_templates:
            self.dollar = DollarRecognizer()
            self._setup_better_templates()
            
        if use_ml:
            self.statistical = StatisticalClassifier()
    
    def _setup_better_templates(self):
        """Setup improved templates for better recognition."""
        # Clear default templates and add better ones
        self.dollar.templates = []
        
        from touchscreen_listener.gestures.dollar_recognizer import Point
        
        # Gesture templates (quick movements)
        gesture_templates = [
            # Quick swipes
            ([Point(0, 0), Point(100, 0)], "gesture_swipe_right"),
            ([Point(0, 0), Point(0, 100)], "gesture_swipe_down"),
            ([Point(0, 0), Point(70, 70)], "gesture_diagonal"),
            ([Point(100, 0), Point(0, 0)], "gesture_swipe_left"),
            ([Point(0, 100), Point(0, 0)], "gesture_swipe_up"),
        ]
        
        # Drawing templates (complex shapes)
        drawing_templates = [
            # Circle
            ([Point(50, 0), Point(70, 10), Point(85, 30), Point(90, 50), 
              Point(85, 70), Point(70, 90), Point(50, 95), Point(30, 90),
              Point(15, 70), Point(10, 50), Point(15, 30), Point(30, 10), Point(50, 0)], 
             "drawing_circle"),
            
            # Square
            ([Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100), Point(0, 0)],
             "drawing_square"),
            
            # Triangle
            ([Point(50, 0), Point(100, 100), Point(0, 100), Point(50, 0)],
             "drawing_triangle"),
            
            # Complex curve
            ([Point(0, 0), Point(20, 10), Point(40, 30), Point(50, 50),
              Point(40, 70), Point(20, 90), Point(0, 100)],
             "drawing_curve"),
        ]
        
        # Add all templates
        for points, name in gesture_templates + drawing_templates:
            self.dollar.add_template(name, points)
    
    def classify_path(self, path: List[Dict[str, float]]) -> Literal['gesture', 'drawing']:
        """
        Classify a path using hybrid approach.
        
        Args:
            path: List of dicts with 'x', 'y', 't' keys
            
        Returns:
            'gesture' or 'drawing'
        """
        if not path or len(path) < 2:
            return 'gesture'
        
        # Calculate basic metrics
        duration = (path[-1]['t'] - path[0]['t']) * 1000  # ms
        point_count = len(path)
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(path)):
            dx = path[i]['x'] - path[i-1]['x']
            dy = path[i]['y'] - path[i-1]['y']
            total_distance += (dx ** 2 + dy ** 2) ** 0.5
        
        # Calculate average velocity
        avg_velocity = VelocityCalculator.calculate_average_velocity(path)
        
        # Simple but effective heuristic based on empirical testing
        # Quick gestures: low duration, high velocity, few points
        # Drawings: high duration, low velocity, many points
        
        # Score-based approach
        gesture_score = 0
        drawing_score = 0
        
        # Duration factor (shorter = more likely gesture)
        if duration < 200:
            gesture_score += 2.0
        elif duration > 800:
            drawing_score += 2.0
        
        # Point count factor (fewer = more likely gesture)
        if point_count < 8:
            gesture_score += 1.5
        elif point_count > 15:
            drawing_score += 1.5
        
        # Velocity factor (faster = more likely gesture)
        if avg_velocity > 100:
            gesture_score += 1.0
        elif avg_velocity < 30:
            drawing_score += 1.0
        
        # Use ML classifier as tie-breaker
        if self.use_ml:
            try:
                ml_result = self.statistical.classify_path(path)
                if ml_result == 'gesture':
                    gesture_score += 0.5
                else:
                    drawing_score += 0.5
            except Exception:
                pass
        
        # Use $1 recognizer as additional factor
        if self.use_templates:
            dollar_result = self.dollar.classify_path(path)
            base_type = dollar_result.split('_')[0]
            if base_type == 'gesture':
                gesture_score += 0.3
            else:
                drawing_score += 0.3
        
        return 'gesture' if gesture_score >= drawing_score else 'drawing'
    
    def _calculate_dollar_confidence(self, path: List[Dict[str, float]]) -> float:
        """Calculate confidence score for $1 recognizer."""
        from touchscreen_listener.gestures.dollar_recognizer import Point
        
        points = [Point(p['x'], p['y']) for p in path]
        
        # Find best matching template
        best_score = float('inf')
        best_template = None
        
        for template in self.dollar.templates:
            # Simple distance calculation
            score = 0
            template_points = template.points
            
            # Resample path to match template length
            if len(points) > 0 and len(template_points) > 0:
                # Simple distance metric
                step = len(points) / len(template_points)
                for i, template_point in enumerate(template_points):
                    idx = min(int(i * step), len(points) - 1)
                    dx = points[idx].x - template_point.x
                    dy = points[idx].y - template_point.y
                    score += (dx ** 2 + dy ** 2) ** 0.5
                
                score /= len(template_points)
                
                if score < best_score:
                    best_score = score
                    best_template = template
        
        # Convert score to confidence (lower score = higher confidence)
        if best_score < 10:
            return 0.9
        elif best_score < 30:
            return 0.7
        elif best_score < 50:
            return 0.5
        else:
            return 0.3
    
    def _calculate_avg_velocity(self, path: List[Dict[str, float]]) -> float:
        """Calculate average velocity of the path."""
        return VelocityCalculator.calculate_average_velocity(path)
    
    def add_training_example(self, path: List[Dict[str, float]], label: str):
        """
        Add a training example to improve the statistical classifier.
        
        Args:
            path: The path data
            label: 'gesture' or 'drawing'
        """
        if self.use_ml:
            try:
                self.statistical.add_training_example(path, label)
            except AttributeError:
                # Statistical classifier might not support online learning
                pass
    
    def get_classifier_info(self) -> Dict[str, any]:
        """Get information about the current classifier state."""
        info = {
            'using_templates': self.use_templates,
            'using_ml': self.use_ml,
        }
        
        if self.use_templates:
            info['template_count'] = len(self.dollar.templates)
            info['template_names'] = [t.name for t in self.dollar.templates]
        
        if self.use_ml:
            try:
                info['ml_trained'] = hasattr(self.statistical, 'is_trained')
                info['training_examples'] = len(self.statistical.training_data) if hasattr(self.statistical, 'training_data') else 'unknown'
            except:
                info['ml_status'] = 'available'
        
        return info


# Convenience function
def classify_path(path: List[Dict[str, float]]) -> Literal['gesture', 'drawing']:
    """Simple interface to classify a path using hybrid approach."""
    classifier = HybridClassifier()
    return classifier.classify_path(path)