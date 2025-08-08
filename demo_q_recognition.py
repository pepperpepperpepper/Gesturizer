#!/usr/bin/env python3
"""$Q Super-Quick Recognition Demo with Visual Feedback.

This demo shows how to use the $Q Super-Quick Recognizer to recognize
gestures and provides visual feedback for the recognition results.
$Q is designed for low-resource devices and provides fast, articulation-invariant
recognition using point clouds and lookup tables.
"""

import json
import os
import sys
from typing import List, Dict, Tuple

import pygame

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from touchscreen_listener.gestures.qdollar_recognizer import (
    QDollarRecognizer,
    classify_path_with_score,
    set_recognition_threshold,
    get_recognition_threshold,
)


class QRecognitionDemo:
    """Interactive demo for $Q gesture recognition."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((2400, 1600))
        pygame.display.set_caption(
            "$Q Super-Quick Recognition Demo - Extra Large Window"
        )

        self.recognizer = QDollarRecognizer()
        self.current_strokes: List[List[Dict]] = []  # List of strokes, each a list of points
        self.is_drawing = False
        self.recognition_result: str | None = None
        self.similarity_score = 0.0
        self.recognition_time = 0.0

        # Set a more lenient recognition threshold for $Q
        set_recognition_threshold(0.3)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.ORANGE = (255, 165, 0)

        # Fonts
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 36)
        self.tiny_font = pygame.font.Font(None, 24)

        # Available $Q gestures (predefined point clouds)
        self.gestures = [
            "T", "N", "D", "P", "X", "H", "I", "exclamation", 
            "line", "five-point star", "null", "arrowhead", 
            "pitchfork", "six-point star", "asterisk", "half-note"]

    def run(self) -> None:
        """Run the demo loop."""
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.start_drawing(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.is_drawing:
                        self.continue_drawing(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.finish_drawing()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.clear_screen()
                    elif event.key == pygame.K_r:
                        self.recognize_gesture()
                    elif event.key == pygame.K_s:
                        self.save_drawing()
                    elif event.key == pygame.K_l:
                        self.load_drawing()
                    elif event.key == pygame.K_UP:
                        self.increase_threshold()
                    elif event.key == pygame.K_DOWN:
                        self.decrease_threshold()
                    elif event.key == pygame.K_d:
                        self.delete_user_gestures()

            self.draw()
            clock.tick(60)

    def start_drawing(self, pos: Tuple[int, int]) -> None:
        """Start a new drawing path."""
        self.current_strokes.append([])  # Start new stroke
        self.is_drawing = True
        self.add_point(pos)

    def continue_drawing(self, pos: Tuple[int, int]) -> None:
        """Add a point while drawing."""
        self.add_point(pos)

    def finish_drawing(self) -> None:
        """Finish drawing and trigger recognition."""
        self.is_drawing = False
        # Recognize only on 'R' key

    def add_point(self, pos: Tuple[int, int]) -> None:
        """Add a point to the current path."""
        x, y = pos
        point = {
            "x": float(x),
            "y": float(y),
            "t": float(pygame.time.get_ticks()),
        }
        if self.current_strokes:
            self.current_strokes[-1].append(point)

    def recognize_gesture(self) -> None:
        """Recognize the drawn gesture using the $Q recognizer."""
        all_points = []
        for stroke_id, stroke in enumerate(self.current_strokes, 1):
            for p in stroke:
                p_copy = p.copy()
                p_copy['stroke_id'] = stroke_id
                all_points.append(p_copy)
        
        if len(all_points) < 2:
            self.recognition_result = "Draw more points!"
            self.similarity_score = 0.0
            self.recognition_time = 0.0
            return
        
        try:
            result = self.recognizer.recognize(all_points)
            self.recognition_result = result.name
            self.similarity_score = result.score
            self.recognition_time = result.time_ms
        except Exception as e:  # pragma: no cover
            self.recognition_result = f"Error: {e}"
            self.similarity_score = 0.0
            self.recognition_time = 0.0

    def clear_screen(self) -> None:
        """Clear the drawing and results."""
        self.current_strokes = []
        self.recognition_result = None
        self.similarity_score = 0.0
        self.recognition_time = 0.0

    def save_drawing(self) -> None:
        """Save the current drawing to a JSON file."""
        if not self.current_strokes:
            return
        data = {
            "strokes": self.current_strokes,
            "recognized_as": self.recognition_result,
            "similarity": self.similarity_score,
            "time_ms": self.recognition_time,
        }
        with open("saved_q_drawing.json", "w") as f:
            json.dump(data, f, indent=2)

    def load_drawing(self) -> None:
        """Load a previously saved drawing."""
        try:
            with open("saved_q_drawing.json", "r") as f:
                data = json.load(f)
            self.current_strokes = data["strokes"]
            self.recognition_result = data["recognized_as"]
            self.similarity_score = data["similarity"]
            self.recognition_time = data.get("time_ms", 0.0)
        except FileNotFoundError:
            self.recognition_result = "No saved drawing found"

    def increase_threshold(self) -> None:
        """Increase the recognition threshold by 0.05 (max 1.0)."""
        current = get_recognition_threshold()
        set_recognition_threshold(min(1.0, current + 0.05))

    def decrease_threshold(self) -> None:
        """Decrease the recognition threshold by 0.05 (min 0.0)."""
        current = get_recognition_threshold()
        set_recognition_threshold(max(0.0, current - 0.05))

    def delete_user_gestures(self) -> None:
        """Delete all user-defined gestures."""
        count = self.recognizer.delete_user_gestures()
        self.recognition_result = f"Deleted {count - 16} user gestures"
        self.similarity_score = 0.0
        self.recognition_time = 0.0

    def draw(self) -> None:
        """Render the UI and current drawing."""
        self.screen.fill(self.WHITE)
        
        # Drawing area boundary
        drawing_area = pygame.Rect(400, 200, 1600, 1000)
        pygame.draw.rect(self.screen, self.GRAY, drawing_area, 3)
        
        # Center crosshair
        cx, cy = 1200, 800
        pygame.draw.line(
            self.screen, (200, 200, 200), (cx - 40, cy), (cx + 40, cy), 2
        )
        pygame.draw.line(
            self.screen, (200, 200, 200), (cx, cy - 40), (cx, cy + 40), 2
        )
        
        # Instructions
        current_threshold = get_recognition_threshold()
        instructions = [
            "$Q Super-Quick Gesture Recognition Demo",
            "Draw gestures in the gray box below:",
            "Draw multi-stroke gestures with multiple mouse drags before pressing R to recognize.",
            "",
            "Available gestures:",
            "• Letters: T, N, D, P, X, H, I",
            "• Symbols: ! (exclamation), - (line), * (asterisk)",
            "• Shapes: ★ (5-point star), ✶ (6-point star), ▭ (rectangle)",
            "• Music: ♫ (half-note)",
            "• Other: ∅ (null), → (arrowhead), ⚡ (pitchfork)",
            "",
            "Controls:",
            "C: Clear   R: Recognize   S: Save   L: Load",
            "UP/DOWN: Adjust threshold   D: Delete user gestures",
            f"Current threshold: {current_threshold:.2f}",
            "$Q is super-fast and articulation-invariant!",
            "Draw at any speed - $Q handles it!",
        ]
        
        y = 10
        for line in instructions:
            if line.startswith("$Q"):
                txt = self.font.render(line, True, self.BLUE)
            elif line.startswith("•"):
                txt = self.small_font.render(line, True, self.BLACK)
            else:
                txt = self.small_font.render(line, True, self.BLACK)
            self.screen.blit(txt, (10, y))
            y += 30
        
        # Area label
        label = self.font.render("DRAWING AREA", True, self.GRAY)
        self.screen.blit(label, (400, 160))
        
        # Available gestures count
        gesture_count = len(self.recognizer.PointClouds)
        count_text = f"Total gestures: {gesture_count} (16 built-in + {gesture_count - 16} custom)"
        self.screen.blit(
            self.small_font.render(count_text, True, self.GRAY), (10, 200)
        )
        
        # Draw all strokes
        if self.current_strokes:
            for stroke in self.current_strokes:
                pts = [(p["x"], p["y"]) for p in stroke]
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, self.RED, False, pts, 6)
                for pt in pts:
                    pygame.draw.circle(
                        self.screen, self.BLUE, (int(pt[0]), int(pt[1])), 8
                    )
        
        # Recognition result
        if self.recognition_result:
            result_text = f"Recognized: {self.recognition_result}"
            score_text = f"Similarity: {self.similarity_score:.2f}"
            time_text = f"Time: {self.recognition_time:.2f}ms"
            
            # Color code based on confidence
            if self.similarity_score > 0.8:
                color = self.GREEN
            elif self.similarity_score > 0.5:
                color = self.ORANGE
            else:
                color = self.RED
            
            self.screen.blit(
                self.font.render(result_text, True, color), (1200, 1300)
            )
            self.screen.blit(
                self.font.render(score_text, True, color), (1200, 1350)
            )
            self.screen.blit(
                self.small_font.render(time_text, True, color), (1200, 1400)
            )
        
        # Performance info
        perf_text = "$Q uses point clouds and lookup tables for super-fast recognition"
        self.screen.blit(
            self.tiny_font.render(perf_text, True, self.GRAY), (10, 1570)
        )
        
        pygame.display.flip()


def main() -> None:
    """Entry point for the demo."""
    demo = QRecognitionDemo()
    try:
        demo.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
