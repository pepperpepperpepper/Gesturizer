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
        if len(points) <= 1:
            return [points[0]] * num_points if points else [Point(0, 0)] * num_points

        total_length = self._path_length(points)
        if total_length == 0:
            return [points[0]] * num_points

        interval = total_length / (num_points - 1)
        D = 0.0
        resampled = [points[0]]
        i = 1
        while i < len(points):
            curr_point = points[i]
            prev_point = points[i-1]
            d = math.sqrt((curr_point.x - prev_point.x) ** 2 + (curr_point.y - prev_point.y) ** 2)
            if d > 0:
                if D + d >= interval:
                    ratio = (interval - D) / d
                    qx = prev_point.x + ratio * (curr_point.x - prev_point.x)
                    qy = prev_point.y + ratio * (curr_point.y - prev_point.y)
                    q = Point(qx, qy)
                    resampled.append(q)
                    points.insert(i, q)  # insert q at i
                    D = 0.0
                else:
                    D += d
            i += 1

        # Add last point if needed due to rounding
        if len(resampled) == num_points - 1:
            resampled.append(points[-1])

        return resampled
    
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
        """Scale points non-uniformly to fit within a square, matching original $1 algorithm."""
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return points
            
        # Non-uniform scaling as per original $1 algorithm
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
        self._load_json_templates()  # Then load any custom JSON templates
    
    def _load_js_templates(self):
        """Load the exact 16 templates from JavaScript $1 recognizer."""
        js_path = os.path.join(os.path.dirname(__file__), '..', '..', 'js_templates.json')
        
        # Try to load JS templates from file
        if os.path.exists(js_path):
            try:
                with open(js_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    templates_data = data.get('templates', data)
                    
                    for template in templates_data:
                        name = template['name']
                        points = [Point(p['x'], p['y']) for p in template['points']]
                        self.add_template(name, points)
                    return
            except Exception as e:
                print(f"Warning: Could not load JS templates from file: {e}")
    
    def _load_json_templates(self):
        """Load templates from JSON file with error handling."""
        json_path = os.path.join(os.path.dirname(__file__), 'templates.json')
        self.load_templates(json_path)
    
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
    
    def add_template(self, name: str, points: List[Point]) -> int:
        """Add a new gesture template, returns count of templates with this name."""
        template = GestureTemplate(name, points)
        self.templates.append(template)
        
        # Return count of templates with this name (like JS AddGesture)
        return sum(1 for t in self.templates if t.name == name)
    
    def add_gesture(self, name: str, points: List[Point]) -> int:
        """Add a new gesture template (alias for add_template to match JS API)."""
        return self.add_template(name, points)
    

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
    
    def classify_with_score(self, path: List[Dict[str, float]], use_protractor: bool = False) -> Tuple[str, float]:
        """
        Classify a path and return both classification and similarity score.
        
        Args:
            path: List of dicts with 'x', 'y', 't' keys
            use_protractor: Whether to use Protractor optimization (vector-based cosine distance)
            
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
        
        print("Calculating scores for templates:")
        for template in self.templates:
            if use_protractor:
                score = self._distance_at_best_angle_protractor(normalized_points, template.points)
            else:
                score = self._distance_at_best_angle(normalized_points, template.points)
            print(f"Template {template.name}: distance {score:.4f}")
            if score < best_score:
                best_score = score
                best_template = template
        print(f"Best template: {best_template.name if best_template else 'None'} with distance {best_score:.4f}")
        
        # Convert distance to similarity score (0.0-1.0)
        similarity_score = self._distance_to_similarity(best_score)
                # Use global config
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
    
    def _distance_to_similarity(self, distance: float, use_protractor: bool = False) -> float:
        """Convert distance to similarity score (0.0-1.0)."""
        if use_protractor:
            return 1.0 - (distance / (math.pi / 2))
        else:
            square_size = 250.0
            half_diagonal = 0.5 * math.sqrt(square_size ** 2 + square_size ** 2)
            return 1.0 - (distance / half_diagonal)
    
    def _distance_at_best_angle(self, points: List[Point], template_points: List[Point]) -> float:
        """Find the best angle match using Golden Section Search."""
        # Correct golden ratio (≈0.618 instead of ≈1.618)
        PHI = 0.5 * (-1.0 + math.sqrt(5.0))
        
        # Search range [-45°, +45°] in radians (matching original)
        theta_a = -math.pi / 4
        theta_b = math.pi / 4
        
        # Convergence threshold (2 degrees in radians)
        threshold = 0.0349
        
        # Golden section search with convergence check
        x1 = PHI * theta_a + (1 - PHI) * theta_b
        f1 = self._distance_at_angle(points, template_points, x1)
        
        x2 = (1 - PHI) * theta_a + PHI * theta_b
        f2 = self._distance_at_angle(points, template_points, x2)
        
        while abs(theta_b - theta_a) > threshold:
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

    def _distance_at_best_angle_protractor(self, points: List[Point], template_points: List[Point]) -> float:
        """Find the best angle match using Protractor optimization (vector-based cosine distance)."""
        # Vectorize both point sets
        vector1 = self._vectorize(points)
        vector2 = self._vectorize(template_points)
        
        # Calculate optimal cosine distance
        distance = self._optimal_cosine_distance(vector1, vector2)
        return distance

    def _vectorize(self, points: List[Point]) -> List[float]:
        """Convert points to a normalized vector representation for Protractor."""
        centroid = self._calculate_centroid(points)
        
        # Create vector by subtracting centroid and flattening
        vector = []
        for point in points:
            vector.append(point.x - centroid.x)
            vector.append(point.y - centroid.y)
        
        # Normalize the vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector

    def _optimal_cosine_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate optimal cosine distance between two vectors."""
        if len(vector1) != len(vector2):
            return float('inf')

        a = 0.0
        b = 0.0
        for i in range(0, len(vector1), 2):
            a += vector1[i] * vector2[i] + vector1[i + 1] * vector2[i + 1]
            b += vector1[i] * vector2[i + 1] - vector1[i + 1] * vector2[i]

        if a == 0.0:
            angle = math.pi / 2 if b > 0 else -math.pi / 2
        else:
            angle = math.atan(b / a)

        opt_dot = a * math.cos(angle) + b * math.sin(angle)
        opt_dot = max(-1.0, min(1.0, opt_dot))
        return math.acos(opt_dot)
    
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
            # Don't clear JS templates, just add new ones (like JS AddGesture)
            for name, points in loaded_templates:
                # Check if template already exists and replace it
                existing_idx = None
                for idx, template in enumerate(self.templates):
                    if template.name == name:
                        existing_idx = idx
                        break
                
                if existing_idx is not None:
                    # Replace existing template
                    self.templates[existing_idx] = GestureTemplate(name, points)
                else:
                    # Add new template
                    self.add_template(name, points)
            print(f"Successfully loaded {len(loaded_templates)} templates from '{filename}'")
        else:
            print(f"Warning: No valid templates found in '{filename}'")


class RecognitionConfig:
    """Configuration for recognition sensitivity."""
    
    def __init__(self):
        self.similarity_threshold = 0.85  # Higher threshold for better accuracy
        self.max_distance = 100.0
        self.resample_points = 64  # Match original $1 algorithm
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
    
    def add_gesture(self, name: str, points: List[Dict[str, float]]) -> int:
        """Add a single gesture template (matches JS API)."""
        gesture_points = [Point(p['x'], p['y']) for p in points]
        return self.recognizer.add_gesture(name, gesture_points)
    
    def delete_user_gestures(self) -> int:
        """Delete all user-defined templates (matches JS API)."""
        return self.recognizer.delete_user_gestures()
    
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


