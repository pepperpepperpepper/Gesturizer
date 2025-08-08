"""
$Q Super-Quick Recognizer Implementation

This implements the $Q Super-Quick Recognizer algorithm for gesture recognition.
It's designed for low-resource devices and provides fast recognition with
articulation-invariant matching using point clouds and lookup tables.

Reference: https://dl.acm.org/citation.cfm?id=3229434.3229465
"""

import math
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..utils.gesture_utils import Point, GeometryUtils


@dataclass
class QResult:
    """Result of $Q recognition with name, score, and timing."""
    name: str
    score: float
    time_ms: float


class QPoint:
    """Point class for $Q recognizer with integer coordinates for LUT indexing."""
    
    def __init__(self, x: float, y: float, stroke_id: int = 1):
        self.X = x
        self.Y = y
        self.ID = stroke_id
        self.IntX = 0  # for indexing into the LUT
        self.IntY = 0  # for indexing into the LUT


class QPointCloud:
    """Point cloud class for $Q recognizer."""
    
    def __init__(self, name: str, points: List[QPoint]):
        self.Name = name
        self.Points = self._resample(points, NumPoints)
        self.Points = self._scale(self.Points)
        self.Points = self._translate_to(self.Points, Origin)
        self.Points = self._make_int_coords(self.Points)  # fills in (IntX, IntY) values
        self.LUT = self._compute_lut(self.Points)
    
    def _resample(self, points: List[QPoint], n: int) -> List[QPoint]:
        """Resample points to have equal spacing."""
        if len(points) <= 1:
            return points
        
        I = self._path_length(points) / (n - 1)  # interval length
        D = 0.0
        newpoints = [points[0]]
        
        i = 1
        while i < len(points):
            if points[i].ID == points[i-1].ID:
                d = self._euclidean_distance(points[i-1], points[i])
                if (D + d) >= I:
                    qx = points[i-1].X + ((I - D) / d) * (points[i].X - points[i-1].X)
                    qy = points[i-1].Y + ((I - D) / d) * (points[i].Y - points[i-1].Y)
                    q = QPoint(qx, qy, points[i].ID)
                    newpoints.append(q)
                    points.insert(i, q)  # insert 'q' at position i
                    D = 0.0
                else:
                    D += d
            i += 1
        
        # sometimes we fall a rounding-error short of adding the last point
        if len(newpoints) == n - 1:
            newpoints.append(QPoint(points[-1].X, points[-1].Y, points[-1].ID))
        
        return newpoints
    
    def _path_length(self, points: List[QPoint]) -> float:
        """Calculate total path length."""
        d = 0.0
        for i in range(1, len(points)):
            if points[i].ID == points[i-1].ID:
                d += self._euclidean_distance(points[i-1], points[i])
        return d
    
    def _euclidean_distance(self, pt1: QPoint, pt2: QPoint) -> float:
        """Calculate Euclidean distance between two points."""
        dx = pt2.X - pt1.X
        dy = pt2.Y - pt1.Y
        return math.sqrt(dx * dx + dy * dy)
    
    def _scale(self, points: List[QPoint]) -> List[QPoint]:
        """Scale points to [0,1] range."""
        min_x = min(p.X for p in points)
        max_x = max(p.X for p in points)
        min_y = min(p.Y for p in points)
        max_y = max(p.Y for p in points)
        
        size = max(max_x - min_x, max_y - min_y)
        if size == 0:
            return points
        
        newpoints = []
        for point in points:
            qx = (point.X - min_x) / size
            qy = (point.Y - min_y) / size
            newpoints.append(QPoint(qx, qy, point.ID))
        
        return newpoints
    
    def _translate_to(self, points: List[QPoint], pt: QPoint) -> List[QPoint]:
        """Translate points' centroid to pt."""
        c = self._centroid(points)
        newpoints = []
        
        for point in points:
            qx = point.X + pt.X - c.X
            qy = point.Y + pt.Y - c.Y
            newpoints.append(QPoint(qx, qy, point.ID))
        
        return newpoints
    
    def _centroid(self, points: List[QPoint]) -> QPoint:
        """Calculate centroid of points."""
        x = sum(p.X for p in points) / len(points)
        y = sum(p.Y for p in points) / len(points)
        return QPoint(x, y, 0)
    
    def _make_int_coords(self, points: List[QPoint]) -> List[QPoint]:
        """Convert to integer coordinates for LUT indexing."""
        for point in points:
            point.IntX = round((point.X + 1.0) / 2.0 * (MaxIntCoord - 1))
            point.IntY = round((point.Y + 1.0) / 2.0 * (MaxIntCoord - 1))
        return points
    
    def _compute_lut(self, points: List[QPoint]) -> List[List[int]]:
        """Compute lookup table for fast distance calculation."""
        LUT = []
        for _ in range(LUTSize):
            LUT.append([0] * LUTSize)
        
        for x in range(LUTSize):
            for y in range(LUTSize):
                u = -1
                b = float('inf')
                for i, point in enumerate(points):
                    row = round(point.IntX / LUTScaleFactor)
                    col = round(point.IntY / LUTScaleFactor)
                    d = ((row - x) ** 2) + ((col - y) ** 2)
                    if d < b:
                        b = d
                        u = i
                LUT[x][y] = u
        
        return LUT


# $Q Recognizer constants
NumPointClouds = 17
NumPoints = 32
Origin = QPoint(0, 0, 0)
MaxIntCoord = 1024  # (IntX, IntY) range from [0, MaxIntCoord - 1]
LUTSize = 64  # default size of the lookup table is 64 x 64
LUTScaleFactor = MaxIntCoord / LUTSize  # used to scale from (IntX, IntY) to LUT


class QDollarRecognizer:
    """$Q Super-Quick Recognizer for gesture classification."""
    
    def __init__(self):
        self.PointClouds = []
        self._load_predefined_clouds()
    
    def _load_predefined_clouds(self):
        """Load the predefined 16 point clouds from the original $Q recognizer."""
        # T
        self.PointClouds.append(QPointCloud("T", [
            QPoint(30,7,1), QPoint(103,7,1),
            QPoint(66,7,2), QPoint(66,87,2)
        ]))
        
        # N
        self.PointClouds.append(QPointCloud("N", [
            QPoint(177,92,1), QPoint(177,2,1),
            QPoint(182,1,2), QPoint(246,95,2),
            QPoint(247,87,3), QPoint(247,1,3)
        ]))
        
        # D
        self.PointClouds.append(QPointCloud("D", [
            QPoint(345,9,1), QPoint(345,87,1),
            QPoint(351,8,2), QPoint(363,8,2), QPoint(372,9,2), QPoint(380,11,2), 
            QPoint(386,14,2), QPoint(391,17,2), QPoint(394,22,2), QPoint(397,28,2), 
            QPoint(399,34,2), QPoint(400,42,2), QPoint(400,50,2), QPoint(400,56,2), 
            QPoint(399,61,2), QPoint(397,66,2), QPoint(394,70,2), QPoint(391,74,2), 
            QPoint(386,78,2), QPoint(382,81,2), QPoint(377,83,2), QPoint(372,85,2), 
            QPoint(367,87,2), QPoint(360,87,2), QPoint(355,88,2), QPoint(349,87,2)
        ]))
        
        # P
        self.PointClouds.append(QPointCloud("P", [
            QPoint(507,8,1), QPoint(507,87,1),
            QPoint(513,7,2), QPoint(528,7,2), QPoint(537,8,2), QPoint(544,10,2), 
            QPoint(550,12,2), QPoint(555,15,2), QPoint(558,18,2), QPoint(560,22,2), 
            QPoint(561,27,2), QPoint(562,33,2), QPoint(561,37,2), QPoint(559,42,2), 
            QPoint(556,45,2), QPoint(550,48,2), QPoint(544,51,2), QPoint(538,53,2), 
            QPoint(532,54,2), QPoint(525,55,2), QPoint(519,55,2), QPoint(513,55,2), 
            QPoint(510,55,2)
        ]))
        
        # X
        self.PointClouds.append(QPointCloud("X", [
            QPoint(30,146,1), QPoint(106,222,1),
            QPoint(30,225,2), QPoint(106,146,2)
        ]))
        
        # H
        self.PointClouds.append(QPointCloud("H", [
            QPoint(188,137,1), QPoint(188,225,1),
            QPoint(188,180,2), QPoint(241,180,2),
            QPoint(241,137,3), QPoint(241,225,3)
        ]))
        
        # I
        self.PointClouds.append(QPointCloud("I", [
            QPoint(371,149,1), QPoint(371,221,1),
            QPoint(341,149,2), QPoint(401,149,2),
            QPoint(341,221,3), QPoint(401,221,3)
        ]))
        
        # exclamation
        self.PointClouds.append(QPointCloud("exclamation", [
            QPoint(526,142,1), QPoint(526,204,1),
            QPoint(526,221,2)
        ]))
        
        # line
        self.PointClouds.append(QPointCloud("line", [
            QPoint(12,347,1), QPoint(119,347,1)
        ]))
        
        # five-point star
        self.PointClouds.append(QPointCloud("five-point star", [
            QPoint(177,396,1), QPoint(223,299,1), QPoint(262,396,1), 
            QPoint(168,332,1), QPoint(278,332,1), QPoint(184,397,1)
        ]))
        
        # null
        self.PointClouds.append(QPointCloud("null", [
            QPoint(382,310,1), QPoint(377,308,1), QPoint(373,307,1), QPoint(366,307,1), 
            QPoint(360,310,1), QPoint(356,313,1), QPoint(353,316,1), QPoint(349,321,1), 
            QPoint(347,326,1), QPoint(344,331,1), QPoint(342,337,1), QPoint(341,343,1), 
            QPoint(341,350,1), QPoint(341,358,1), QPoint(342,362,1), QPoint(344,366,1), 
            QPoint(347,370,1), QPoint(351,374,1), QPoint(356,379,1), QPoint(361,382,1), 
            QPoint(368,385,1), QPoint(374,387,1), QPoint(381,387,1), QPoint(390,387,1), 
            QPoint(397,385,1), QPoint(404,382,1), QPoint(408,378,1), QPoint(412,373,1), 
            QPoint(416,367,1), QPoint(418,361,1), QPoint(419,353,1), QPoint(418,346,1), 
            QPoint(417,341,1), QPoint(416,336,1), QPoint(413,331,1), QPoint(410,326,1), 
            QPoint(404,320,1), QPoint(400,317,1), QPoint(393,313,1), QPoint(392,312,1),
            QPoint(418,309,2), QPoint(337,390,2)
        ]))
        
        # arrowhead
        self.PointClouds.append(QPointCloud("arrowhead", [
            QPoint(506,349,1), QPoint(574,349,1),
            QPoint(525,306,2), QPoint(584,349,2), QPoint(525,388,2)
        ]))
        
        # pitchfork
        self.PointClouds.append(QPointCloud("pitchfork", [
            QPoint(38,470,1), QPoint(36,476,1), QPoint(36,482,1), QPoint(37,489,1), 
            QPoint(39,496,1), QPoint(42,500,1), QPoint(46,503,1), QPoint(50,507,1), 
            QPoint(56,509,1), QPoint(63,509,1), QPoint(70,508,1), QPoint(75,506,1), 
            QPoint(79,503,1), QPoint(82,499,1), QPoint(85,493,1), QPoint(87,487,1), 
            QPoint(88,480,1), QPoint(88,474,1), QPoint(87,468,1),
            QPoint(62,464,2), QPoint(62,571,2)
        ]))
        
        # six-point star
        self.PointClouds.append(QPointCloud("six-point star", [
            QPoint(177,554,1), QPoint(223,476,1), QPoint(268,554,1), QPoint(183,554,1),
            QPoint(177,490,2), QPoint(223,568,2), QPoint(268,490,2), QPoint(183,490,2)
        ]))
        
        # asterisk
        self.PointClouds.append(QPointCloud("asterisk", [
            QPoint(325,499,1), QPoint(417,557,1),
            QPoint(417,499,2), QPoint(325,557,2),
            QPoint(371,486,3), QPoint(371,571,3)
        ]))
        
        # half-note
        self.PointClouds.append(QPointCloud("half-note", [
            QPoint(546,465,1), QPoint(546,531,1),
            QPoint(540,530,2), QPoint(536,529,2), QPoint(533,528,2), QPoint(529,529,2), 
            QPoint(524,530,2), QPoint(520,532,2), QPoint(515,535,2), QPoint(511,539,2), 
            QPoint(508,545,2), QPoint(506,548,2), QPoint(506,554,2), QPoint(509,558,2), 
            QPoint(512,561,2), QPoint(517,564,2), QPoint(521,564,2), QPoint(527,563,2), 
            QPoint(531,560,2), QPoint(535,557,2), QPoint(538,553,2), QPoint(542,548,2), 
            QPoint(544,544,2), QPoint(546,540,2), QPoint(546,536,2)
        ]))
    
    def recognize(self, points: List[Dict[str, Any]]) -> QResult:
        """
        Recognize a gesture using the $Q recognizer.
        
        Args:
            points: List of dicts with 'x', 'y', 't' keys
            
        Returns:
            QResult with name, score, and timing
        """
        import time
        t0 = time.time() * 1000  # milliseconds
        
        # Convert to QPoint format
        qpoints = [QPoint(p['x'], p['y'], int(p.get('stroke_id', 1))) for p in points]
        candidate = QPointCloud("", qpoints)
        
        u = -1
        b = float('inf')
        for i, template in enumerate(self.PointClouds):
            d = self._cloud_match(candidate, template, b)
            if d < b:
                b = d  # best (least) distance
                u = i  # point-cloud index
        
        t1 = time.time() * 1000
        
        if u == -1:
            return QResult("No match.", 0.0, t1 - t0)
        else:
            # Convert distance to similarity score (higher is better)
            score = 1.0 / b if b > 1.0 else 1.0
            return QResult(self.PointClouds[u].Name, score, t1 - t0)
    
    def classify_path(self, path: List[Dict[str, Any]]) -> str:
        """Classify a path and return the gesture name."""
        result = self.recognize(path)
        return result.name
    
    def classify_with_score(self, path: List[Dict[str, float]]) -> Tuple[str, float]:
        """Classify a path and return both name and score."""
        result = self.recognize(path)
        return result.name, result.score
    
    def add_gesture(self, name: str, points: List[Dict[str, Any]]) -> int:
        """Add a new gesture template."""
        qpoints = [QPoint(p['x'], p['y'], int(p.get('stroke_id', 1))) for p in points]
        self.PointClouds.append(QPointCloud(name, qpoints))
        
        # Return count of gestures with this name
        num = 0
        for cloud in self.PointClouds:
            if cloud.Name == name:
                num += 1
        return num
    
    def delete_user_gestures(self) -> int:
        """Delete all user-defined gestures, keeping only the original 16."""
        self.PointClouds = self.PointClouds[:NumPointClouds]
        return NumPointClouds
    
    def _cloud_match(self, candidate: QPointCloud, template: QPointCloud, min_so_far: float) -> float:
        """Match two point clouds using $Q algorithm."""
        n = len(candidate.Points)
        step = int(math.pow(n, 0.5))
        
        LB1 = self._compute_lower_bound(candidate.Points, template.Points, step, template.LUT)
        LB2 = self._compute_lower_bound(template.Points, candidate.Points, step, candidate.LUT)
        
        for i in range(0, n, step):
            j = i // step
            if j < len(LB1) and LB1[j] < min_so_far:
                min_so_far = min(min_so_far, self._cloud_distance(candidate.Points, template.Points, i, min_so_far))
            if j < len(LB2) and LB2[j] < min_so_far:
                min_so_far = min(min_so_far, self._cloud_distance(template.Points, candidate.Points, i, min_so_far))
        
        return min_so_far
    
    def _cloud_distance(self, pts1: List[QPoint], pts2: List[QPoint], start: int, min_so_far: float) -> float:
        """Calculate distance between two point clouds."""
        n = len(pts1)
        unmatched = list(range(n))  # indices for pts2 that are not matched
        i = start  # start matching with point 'start' from pts1
        weight = n  # weights decrease from n to 1
        sum_dist = 0.0  # sum distance between the two clouds
        
        while True:
            u = -1
            b = float('inf')
            best_idx = -1
            for idx, j in enumerate(unmatched):
                d = self._sqr_euclidean_distance(pts1[i], pts2[j])
                if d < b:
                    b = d
                    u = j
                    best_idx = idx
            
            if u == -1 or best_idx == -1:
                break  # No more matches possible
            
            unmatched.pop(best_idx)  # remove item at best_idx
            sum_dist += weight * b
            if sum_dist >= min_so_far:
                return sum_dist  # early abandoning
            
            weight -= 1
            i = (i + 1) % n
            if i == start or not unmatched:
                break
        
        return sum_dist
    
    def _compute_lower_bound(self, pts1: List[QPoint], pts2: List[QPoint], step: int, LUT: List[List[int]]) -> List[float]:
        """Compute lower bound for early abandoning."""
        n = len(pts1)
        LB = [0.0] * ((n // step) + 1)
        SAT = [0.0] * n
        LB[0] = 0.0
        
        for i in range(n):
            x = round(pts1[i].IntX / LUTScaleFactor)
            y = round(pts1[i].IntY / LUTScaleFactor)
            index = LUT[x][y]
            d = self._sqr_euclidean_distance(pts1[i], pts2[index])
            SAT[i] = d if i == 0 else SAT[i - 1] + d
            LB[0] += (n - i) * d
        
        for i in range(step, n, step):
            j = i // step
            if j < len(LB):
                LB[j] = LB[0] + i * SAT[n-1] - n * SAT[i-1]
        
        return LB
    
    def _sqr_euclidean_distance(self, pt1: QPoint, pt2: QPoint) -> float:
        """Calculate squared Euclidean distance between two points."""
        dx = pt2.X - pt1.X
        dy = pt2.Y - pt1.Y
        return dx * dx + dy * dy


# Global configuration for recognition sensitivity
class QRecognitionConfig:
    """Configuration for $Q recognition sensitivity."""
    
    def __init__(self):
        self.similarity_threshold = 0.5  # Lower threshold for $Q since scores can be higher
        self.max_distance = 100.0
    
    def set_threshold(self, threshold: float):
        """Set the similarity threshold (0.0-1.0)."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
    
    def get_threshold(self) -> float:
        """Get the current similarity threshold."""
        return self.similarity_threshold


# Global instances
qrecognizer = QDollarRecognizer()
qconfig = QRecognitionConfig()


def classify_path(path: List[Dict[str, float]]) -> str:
    """Classify a path using the $Q Recognizer."""
    return qrecognizer.classify_path(path)


def classify_path_with_score(path: List[Dict[str, float]]) -> Tuple[str, float]:
    """Classify a path and return both classification and similarity score."""
    result = qrecognizer.recognize(path)
    return result.name, result.score


def add_gesture_template(name: str, path: List[Dict[str, float]]):
    """Add a new gesture template."""
    qrecognizer.add_gesture(name, path)


def set_recognition_threshold(threshold: float):
    """Set the recognition similarity threshold."""
    qconfig.set_threshold(threshold)


def get_recognition_threshold() -> float:
    """Get the current recognition similarity threshold."""
    return qconfig.get_threshold()
