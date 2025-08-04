"""
Machine Learning Classifier for Gesture vs Drawing Classification

Uses scikit-learn to train a classifier on extracted features from touch paths.
This is much more robust than hand-coded thresholds.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from ..utils.gesture_utils import VelocityCalculator


class MLPathClassifier:
    """
    ML-based classifier for distinguishing gestures from drawings.
    
    Features extracted:
    - Path length / bounding box ratio (compactness)
    - Average velocity
    - Velocity variance
    - Number of direction changes
    - Total duration
    - Point density
    """
    
    def __init__(self, model_path: str = "gesture_classifier.pkl"):
        self.model_path = model_path
        self.classifier = None
        self.scaler = StandardScaler()
        self._load_or_train_model()
    
    def _extract_features(self, path: List[Dict[str, float]]) -> np.ndarray:
        """Extract features from a path for ML classification."""
        if len(path) < 2:
            return np.array([0, 0, 0, 0, 0, 0])
        
        # Convert to numpy arrays
        x_coords = np.array([p['x'] for p in path])
        y_coords = np.array([p['y'] for p in path])
        times = np.array([p['t'] for p in path])
        
        # Basic geometric features
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        width = max_x - min_x
        height = max_y - min_y
        
        # Path length
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # Bounding box area
        bbox_area = width * height if width > 0 and height > 0 else 1.0
        
        # Compactness (path length / bounding box diagonal)
        diagonal = np.sqrt(width**2 + height**2)
        compactness = path_length / diagonal if diagonal > 0 else 1.0
        
        # Velocity features
        dt = np.diff(times)
        dt = np.where(dt == 0, 0.001, dt)  # Avoid division by zero
        velocities = np.sqrt(dx**2 + dy**2) / dt
        
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        velocity_variance = np.var(velocities) if len(velocities) > 0 else 0
        
        # Direction changes
        if len(dx) > 2:
            angles = np.arctan2(dy, dx)
            angle_diffs = np.diff(angles)
            direction_changes = np.sum(np.abs(angle_diffs) > np.pi/2)
        else:
            direction_changes = 0
        
        # Duration and density
        duration = times[-1] - times[0]
        point_density = len(path) / duration if duration > 0 else 0
        
        return np.array([
            compactness,
            avg_velocity,
            velocity_variance,
            direction_changes,
            duration,
            point_density
        ])
    
    def _generate_training_data(self):
        """Generate synthetic training data for demonstration."""
        # This would normally be real user data
        np.random.seed(42)
        
        # Gesture examples (quick, straight movements)
        gesture_features = []
        for _ in range(50):
            # Quick straight line
            length = np.random.uniform(50, 200)
            duration = np.random.uniform(0.1, 0.3)
            points = np.random.randint(5, 15)
            
            features = np.array([
                np.random.uniform(1.0, 1.5),  # compactness
                length / duration,  # velocity
                np.random.uniform(0.1, 1.0),  # velocity variance
                np.random.randint(0, 2),  # direction changes
                duration,
                points / duration  # point density
            ])
            gesture_features.append(features)
        
        # Drawing examples (slow, complex movements)
        drawing_features = []
        for _ in range(50):
            # Slow complex shape
            length = np.random.uniform(100, 500)
            duration = np.random.uniform(1.0, 5.0)
            points = np.random.randint(20, 100)
            
            features = np.array([
                np.random.uniform(2.0, 5.0),  # compactness
                length / duration,  # velocity
                np.random.uniform(0.5, 3.0),  # velocity variance
                np.random.randint(3, 10),  # direction changes
                duration,
                points / duration  # point density
            ])
            drawing_features.append(features)
        
        X = np.vstack([gesture_features, drawing_features])
        y = np.array(['gesture'] * 50 + ['drawing'] * 50)
        
        return X, y
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                return
            except Exception:
                pass  # Re-train if loading fails
        
        # Train new model
        print("Training new gesture classifier...")
        X, y = self._generate_training_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_scaled, y)
        
        # Save model
        self._save_model()
    
    def _save_model(self):
        """Save the trained model."""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler
        }
        joblib.dump(model_data, self.model_path)
    
    def classify_path(self, path: List[Dict[str, float]]) -> str:
        """Classify a path as 'gesture' or 'drawing'."""
        if self.classifier is None:
            return 'gesture'  # Fallback
        
        features = self._extract_features(path)
        features_scaled = self.scaler.transform([features])
        prediction = self.classifier.predict(features_scaled)[0]
        
        return prediction
    
    def get_confidence(self, path: List[Dict[str, float]]) -> float:
        """Get classification confidence."""
        if self.classifier is None:
            return 0.5
        
        features = self._extract_features(path)
        features_scaled = self.scaler.transform([features])
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        return max(probabilities)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return [
            'compactness',
            'avg_velocity',
            'velocity_variance',
            'direction_changes',
            'duration',
            'point_density'
        ]


# Global instance
_classifier = None

def classify_path(path: List[Dict[str, float]]) -> str:
    """Global function to classify a path."""
    global _classifier
    if _classifier is None:
        _classifier = MLPathClassifier()
    return _classifier.classify_path(path)


def get_classification_confidence(path: List[Dict[str, float]]) -> float:
    """Get classification confidence."""
    global _classifier
    if _classifier is None:
        _classifier = MLPathClassifier()
    return _classifier.get_confidence(path)