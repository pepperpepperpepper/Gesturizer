"""
$1 Unistroke Recognizer Implementation

This implements the $1 Unistroke Recognizer algorithm for gesture recognition.
It's designed to distinguish between quick gestures and deliberate drawing
by comparing against known gesture templates.

Reference: https://depts.washington.edu/acelab/proj/dollar/index.html
"""

import math
from typing import List, Dict, Tuple
import json
import os
from ..utils.gesture_utils import Point, GeometryUtils, VelocityCalculator





class GestureTemplate:
    """Represents a gesture template with name and points."""
    def __init__(self, name: str, points: List[Point]):
        self.name = name
        self.points = self._resample_points(points)
        self.points = self._rotate_to_zero(self.points)
        self.points = self._scale_to_square(self.points)
        self.points = self._translate_to_origin(self.points)
    
    def _resample_points(self, points: List[Point], num_points: int = 64) -> List[Point]:
        """Resample points to have equal spacing."""
        if len(points) < 2:
            # Handle edge case: single point or empty
            if len(points) == 1:
                return [points[0]] * num_points
            else:
                # Empty case - return center point
                return [Point(0, 0)] * num_points
            
        total_length = self._path_length(points)
        if total_length == 0:
            # Handle zero-length paths (all points are the same)
            if points:
                return [points[0]] * num_points
            else:
                return [Point(0, 0)] * num_points
            
        interval = total_length / (num_points - 1) if num_points > 1 else 0
        resampled = [points[0]]
        current_distance = 0.0
        
        # Create a copy of points to avoid modifying original
        remaining_points = points[1:]
        
        # Handle edge case where we have only 2 points but need more samples
        if len(remaining_points) == 1 and len(points) == 2:
            # Linear interpolation between the two points
            p1, p2 = points[0], points[1]
            for i in range(1, num_points):
                ratio = i / (num_points - 1)
                new_x = p1.x + ratio * (p2.x - p1.x)
                new_y = p1.y + ratio * (p2.y - p1.y)
                resampled.append(Point(new_x, new_y))
            return resampled
        
        while len(resampled) < num_points and remaining_points:
            prev_point = resampled[-1]
            curr_point = remaining_points[0]
            
            distance = math.sqrt((curr_point.x - prev_point.x) ** 2 + 
                             (curr_point.y - prev_point.y) ** 2)
            
            # Handle zero distance between consecutive points
            if distance == 0:
                remaining_points.pop(0)
                continue
                
            if current_distance + distance >= interval:
                # Calculate new point
                ratio = (interval - current_distance) / distance
                new_x = prev_point.x + ratio * (curr_point.x - prev_point.x)
                new_y = prev_point.y + ratio * (curr_point.y - prev_point.y)
                resampled.append(Point(new_x, new_y))
                current_distance = 0.0
            else:
                current_distance += distance
                remaining_points.pop(0)
        
        # Fill remaining points with the last point if needed
        while len(resampled) < num_points:
            resampled.append(resampled[-1] if resampled else points[-1])
            
        return resampled[:num_points]
    
    def _path_length(self, points: List[Point]) -> float:
        """Calculate total path length."""
        return GeometryUtils.calculate_path_length(points)
    
    def _rotate_to_zero(self, points: List[Point]) -> List[Point]:
        """Rotate points so the first point is at 0 degrees."""
        if len(points) < 2:
            return points
            
        centroid = self._calculate_centroid(points)
        angle = math.atan2(centroid.y - points[0].y, centroid.x - points[0].x)
        return self._rotate_points(points, -angle)
    
    def _calculate_centroid(self, points: List[Point]) -> Point:
        """Calculate the centroid of points."""
        return GeometryUtils.calculate_centroid(points)
    
    def _rotate_points(self, points: List[Point], angle: float) -> List[Point]:
        """Rotate points around the centroid."""
        return GeometryUtils.rotate_points(points, angle)
    
    def _scale_to_square(self, points: List[Point], size: float = 250.0) -> List[Point]:
        """Scale points to fit within a square."""
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return points
            
        scaled = []
        for point in points:
            new_x = point.x * (size / width)
            new_y = point.y * (size / height)
            scaled.append(Point(new_x, new_y))
        
        return scaled
    
    def _translate_to_origin(self, points: List[Point]) -> List[Point]:
        """Translate points so centroid is at origin."""
        centroid = self._calculate_centroid(points)
        translated = []
        
        for point in points:
            translated.append(Point(point.x - centroid.x, point.y - centroid.y))
        
        return translated


class DollarRecognizer:
    """$1 Unistroke Recognizer for gesture classification."""
    
    def __init__(self):
        self.templates: List[GestureTemplate] = []
        # Use global config instead of instance threshold
        self._load_default_templates()
        self._load_json_templates()
        self._add_geometric_templates()
    
    def _load_default_templates(self):
        """Load default gesture templates including geometric shapes."""
        # Simple gesture templates
        templates = {
            'gesture': [
                # Quick swipe right
                [Point(0, 0), Point(50, 0), Point(100, 0)],
                # Quick swipe down
                [Point(0, 0), Point(0, 50), Point(0, 100)],
                # Quick diagonal
                [Point(0, 0), Point(50, 50), Point(100, 100)],
                # Quick circle-like gesture
                [Point(0, 0), Point(10, 10), Point(20, 5), Point(30, 0), Point(25, -10), Point(15, -15), Point(5, -10), Point(0, 0)],
            ],
            'drawing': [
                # Slow circle (more points, smoother)
                [Point(0, 0), Point(5, 2), Point(10, 3), Point(15, 3), Point(20, 2), Point(25, 0), Point(27, -5), Point(25, -10), Point(20, -15), Point(15, -17), Point(10, -15), Point(5, -10), Point(0, 0)],
                # Complex shape
                [Point(0, 0), Point(10, 5), Point(20, 15), Point(30, 25), Point(25, 35), Point(15, 30), Point(5, 20), Point(0, 10)],
            ]
        }
        
        for name, gesture_list in templates.items():
            for i, points in enumerate(gesture_list):
                self.add_template(f"{name}_{i}", points)
    
    def _load_json_templates(self):
        """Load templates from JSON file with error handling."""
        json_path = os.path.join(os.path.dirname(__file__), 'templates.json')
        self.load_templates(json_path)
    
    def _add_geometric_templates(self):
        """Add geometric shape templates (triangle, circle, rectangle, star, heart)."""
        geometric_templates = {
            'triangle': [
                Point(0, -50), Point(-43, 25), Point(43, 25), Point(0, -50)
            ],
            'circle': self._generate_circle_points(50, 64),
            'rectangle': [
                Point(-50, -25), Point(50, -25), Point(50, 25), Point(-50, 25), Point(-50, -25)
            ],
            'star': self._generate_star_points(5, 50, 25),
            'heart': self._generate_heart_points(),
            'arrow': [
                Point(-50, 0), Point(0, -30), Point(20, -15), Point(20, -30), Point(50, 0), 
                Point(20, 30), Point(20, 15), Point(0, 30), Point(-50, 0)
            ],
            'checkmark': [
                Point(-30, 0), Point(-10, 20), Point(30, -20)
            ]
        }
        
        for name, points in geometric_templates.items():
            self.add_template(name, points)
    
    def _generate_circle_points(self, radius: float, num_points: int) -> List[Point]:
        """Generate points for a circle."""
        points = []
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(Point(x, y))
        return points
    
    def _generate_star_points(self, points: int, outer_radius: float, inner_radius: float) -> List[Point]:
        """Generate points for a star."""
        star_points = []
        for i in range(points * 2 + 1):
            angle = math.pi * i / points
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = radius * math.cos(angle - math.pi / 2)
            y = radius * math.sin(angle - math.pi / 2)
            star_points.append(Point(x, y))
        return star_points
    
    def _generate_heart_points(self) -> List[Point]:
        """Generate points for a heart shape."""
        heart_points = []
        for t in [i * 0.1 for i in range(63)]:  # 0 to 2π
            x = 16 * math.sin(t) ** 3
            y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
            heart_points.append(Point(x * 2, y * 2))
        return heart_points
    
    def add_template(self, name: str, points: List[Point]):
        """Add a new gesture template."""
        template = GestureTemplate(name, points)
        self.templates.append(template)
    
    def classify_path(self, path: List[Dict[str, float]]) -> str:
        """
        Classify a path using the $1 Recognizer.
        
        Args:
            path: List of dicts with 'x', 'y', 't' keys
            
        Returns:
            The recognized shape name or 'gesture'/'drawing' for generic classification
        """
        result, _ = self.classify_with_score(path)
        return result
    
    def classify_with_score(self, path: List[Dict[str, float]]) -> Tuple[str, float]:
        """
        Classify a path and return both classification and similarity score.
        
        Args:
            path: List of dicts with 'x', 'y', 't' keys
            
        Returns:
            Tuple of (classification, similarity_score)
        """
        if not path or len(path) < 2:
            return 'gesture', 0.0
        
        # Convert to Points and normalize
        points = [Point(p['x'], p['y']) for p in path]
        
        # Create a temporary template to normalize the input points
        temp_template = GestureTemplate("temp", points)
        normalized_points = temp_template.points
        
        # Find best matching template
        best_score = float('inf')
        best_template = None
        
        for template in self.templates:
            score = self._distance_at_best_angle(normalized_points, template.points)
            if score < best_score:
                best_score = score
                best_template = template
        
        # Convert distance to similarity score (0.0-1.0)
        similarity_score = self._distance_to_similarity(best_score)
        
        if best_template and similarity_score >= config.similarity_threshold:
            # Return the actual template name for geometric shapes
            if best_template.name in ['triangle', 'circle', 'rectangle', 'star', 'heart', 'arrow', 'checkmark']:
                return best_template.name, similarity_score
            else:
                base_type = best_template.name.split('_')[0]
                return base_type, similarity_score
        
        # For debugging: return the best match even if below threshold
        if best_template:
            if best_template.name in ['triangle', 'circle', 'rectangle', 'star', 'heart', 'arrow', 'checkmark']:
                return f"{best_template.name} (low confidence)", similarity_score
            else:
                base_type = best_template.name.split('_')[0]
                return f"{base_type} (low confidence)", similarity_score
        
        # Fallback classification
        fallback_type = self._simple_classification(path)
        return fallback_type, 0.5  # Medium confidence for fallback
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score (0.0-1.0)."""
        # Map distance [0, 50] to similarity [1.0, 0.0] - more sensitive
        max_distance = 50.0
        similarity = max(0.0, 1.0 - (distance / max_distance))
        return min(1.0, max(0.0, similarity))
    
    def _distance_at_best_angle(self, points: List[Point], template_points: List[Point]) -> float:
        """Find the best angle match using Golden Section Search."""
        # Golden ratio
        PHI = (1 + math.sqrt(5)) / 2
        
        # Search range [-45°, +45°] in radians
        theta_a = -math.pi / 4
        theta_b = math.pi / 4
        
        # Golden section search
        x1 = PHI * theta_a + (1 - PHI) * theta_b
        f1 = self._distance_at_angle(points, template_points, x1)
        
        x2 = (1 - PHI) * theta_a + PHI * theta_b
        f2 = self._distance_at_angle(points, template_points, x2)
        
        for _ in range(15):  # 15 iterations for good precision
            if f1 < f2:
                theta_b = x2
                x2 = x1
                f2 = f1
                x1 = PHI * theta_a + (1 - PHI) * theta_b
                f1 = self._distance_at_angle(points, template_points, x1)
            else:
                theta_a = x1
                x1 = x2
                f1 = f2
                x2 = (1 - PHI) * theta_a + PHI * theta_b
                f2 = self._distance_at_angle(points, template_points, x2)
        
        return min(f1, f2)
    
    def _distance_at_angle(self, points: List[Point], template_points: List[Point], angle: float) -> float:
        """Calculate distance at a specific angle."""
        rotated_points = self._rotate_points(points, angle)
        return self._distance(rotated_points, template_points)
    
    def _rotate_points(self, points: List[Point], angle: float) -> List[Point]:
        """Rotate points around the centroid."""
        return GeometryUtils.rotate_points(points, angle)
    
    def _calculate_centroid(self, points: List[Point]) -> Point:
        """Calculate the centroid of points."""
        return GeometryUtils.calculate_centroid(points)

    def _distance(self, points1: List[Point], points2: List[Point]) -> float:
        """Calculate Euclidean distance between two point sets."""
        if len(points1) != len(points2):
            return float('inf')
        
        # Euclidean distance calculation
        distance = 0.0
        for p1, p2 in zip(points1, points2):
            distance += math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        
        return distance / len(points1)
    
    def _simple_classification(self, path: List[Dict[str, float]]) -> str:
        """Simple fallback classification."""
        if len(path) < 10:
            return 'gesture'
        
        avg_velocity = VelocityCalculator.calculate_average_velocity(path)
        if avg_velocity > 100:  # Fast gesture
            return 'gesture'
        
        return 'drawing'
    
    def save_templates(self, filename: str):
        """Save templates to file."""
        data = []
        for template in self.templates:
            points_data = [{'x': p.x, 'y': p.y} for p in template.points]
            data.append({
                'name': template.name,
                'points': points_data
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_templates(self, filename: str):
        """Load templates from file with comprehensive error handling and validation."""
        if not os.path.exists(filename):
            print(f"Warning: Template file '{filename}' not found")
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, IOError) as e:
            print(f"Error loading templates from '{filename}': {e}")
            return
        
        # Handle both old format (list) and new format (dict with 'templates' key)
        templates_data = data.get('templates', data) if isinstance(data, dict) else data
        
        if not isinstance(templates_data, list):
            print(f"Error: Invalid template format in '{filename}'. Expected list of templates.")
            return
        
        loaded_templates = []
        for i, item in enumerate(templates_data):
            try:
                # Validate template structure
                if not isinstance(item, dict):
                    print(f"Warning: Skipping template {i}: not a dictionary")
                    continue
                
                # Validate required fields
                if 'name' not in item:
                    print(f"Warning: Skipping template {i}: missing 'name' field")
                    continue
                
                if 'points' not in item:
                    print(f"Warning: Skipping template {i}: missing 'points' field")
                    continue
                
                name = str(item['name']).strip()
                if not name:
                    print(f"Warning: Skipping template {i}: empty name")
                    continue
                
                points_data = item['points']
                if not isinstance(points_data, list):
                    print(f"Warning: Skipping template '{name}': 'points' must be a list")
                    continue
                
                if len(points_data) < 2:
                    print(f"Warning: Skipping template '{name}': need at least 2 points")
                    continue
                
                # Validate and parse points
                points = []
                valid_points = True
                for j, point_data in enumerate(points_data):
                    if not isinstance(point_data, dict):
                        print(f"Warning: Skipping template '{name}': point {j} is not a dictionary")
                        valid_points = False
                        break
                    
                    try:
                        x = float(point_data.get('x', 0))
                        y = float(point_data.get('y', 0))
                        points.append(Point(x, y))
                    except (ValueError, TypeError):
                        print(f"Warning: Skipping template '{name}': invalid coordinates in point {j}")
                        valid_points = False
                        break
                
                if valid_points and points:
                    loaded_templates.append((name, points))
                    
            except Exception as e:
                print(f"Warning: Error processing template {i}: {e}")
                continue
        
        # Only update templates if we successfully loaded some
        if loaded_templates:
            self.templates = []  # Clear existing templates
            for name, points in loaded_templates:
                self.add_template(name, points)
            print(f"Successfully loaded {len(loaded_templates)} templates from '{filename}'")
        else:
            print(f"Warning: No valid templates found in '{filename}'")


class RecognitionConfig:
    """Configuration for recognition sensitivity."""
    
    def __init__(self):
        self.similarity_threshold = 0.65  # Lowered for easier recognition
        self.max_distance = 100.0
        self.golden_section_iterations = 15
        self.resample_points = 64
        self.rotation_range = math.pi / 4  # 45 degrees
    
    def set_threshold(self, threshold: float):
        """Set the similarity threshold (0.0-1.0)."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
    
    def get_threshold(self) -> float:
        """Get the current similarity threshold."""
        return self.similarity_threshold


class TemplateTrainer:
    """Interface for training custom gesture templates."""
    
    def __init__(self, recognizer: DollarRecognizer):
        self.recognizer = recognizer
        self.training_data = []
    
    def add_training_sample(self, name: str, path: List[Dict[str, float]]):
        """Add a training sample for a gesture."""
        self.training_data.append({
            'name': name,
            'path': path
        })
    
    def train_template(self, name: str, samples: List[List[Dict[str, float]]]) -> bool:
        """Train a new template from multiple samples."""
        if not samples:
            return False
        
        # Average the samples to create a robust template
        all_points = []
        for sample in samples:
            points = [Point(p['x'], p['y']) for p in sample]
            template = GestureTemplate(name, points)
            all_points.append(template.points)
        
        # Average corresponding points
        num_points = len(all_points[0])
        averaged_points = []
        
        for i in range(num_points):
            avg_x = sum(points[i].x for points in all_points) / len(all_points)
            avg_y = sum(points[i].y for points in all_points) / len(all_points)
            averaged_points.append(Point(avg_x, avg_y))
        
        self.recognizer.add_template(name, averaged_points)
        return True
    
    def save_training_data(self, filename: str):
        """Save training data to file."""
        with open(filename, 'w') as f:
            json.dump(self.training_data, f, indent=2)
    
    def load_training_data(self, filename: str):
        """Load training data from file."""
        if not os.path.exists(filename):
            return
        
        with open(filename, 'r') as f:
            self.training_data = json.load(f)


# Convenience functions
recognizer = DollarRecognizer()
config = RecognitionConfig()


def classify_path(path: List[Dict[str, float]]) -> str:
    """Classify a path using the $1 Recognizer."""
    return recognizer.classify_path(path)


def classify_path_with_score(path: List[Dict[str, float]]) -> Tuple[str, float]:
    """Classify a path and return both classification and similarity score."""
    return recognizer.classify_with_score(path)


def add_gesture_template(name: str, path: List[Dict[str, float]]):
    """Add a new gesture template."""
    from touchscreen_listener.gestures.dollar_recognizer import Point
    points = [Point(p['x'], p['y']) for p in path]
    recognizer.add_template(name, points)


def set_recognition_threshold(threshold: float):
    """Set the recognition similarity threshold."""
    config.set_threshold(threshold)


def get_recognition_threshold() -> float:
    """Get the current recognition similarity threshold."""
    return config.get_threshold()


