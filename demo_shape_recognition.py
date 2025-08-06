#!/usr/bin/env python3
"""Shape Recognition Demo with Visual Feedback.

This demo shows how to use the enhanced $1 Recognizer to recognize
geometric shapes and provides visual feedback for the
recognition results.
"""

import json
import os
import sys
from typing import List, Dict, Tuple

import pygame

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from touchscreen_listener.gestures.dollar_recognizer import (
    DollarRecognizer,
    Point,
    classify_path_with_score,
    set_recognition_threshold,
    get_recognition_threshold,
)


class ShapeRecognitionDemo:
    """Interactive demo for shape recognition."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((2400, 1600))
        pygame.display.set_caption(
            "$1 Shape Recognition Demo - Extra Large Window"
        )

        self.recognizer = DollarRecognizer()
        self.current_path: List[Dict] = []
        self.is_drawing = False
        self.recognition_result: str | None = None
        self.similarity_score = 0.0

        # Set a more lenient recognition threshold
        set_recognition_threshold(0.75)

        # Hardcoded upright triangle test (placeholder)
        print(
            "Test upright triangle recognition:",
            self.recognition_result,
            f"score: {self.similarity_score:.2f}",
        )

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)

        # Fonts
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 36)

        # Available shapes (canonical $1 shapes)
        self.shapes = ["triangle", "circle", "rectangle"]

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
                        self.recognize_shape()
                    elif event.key == pygame.K_s:
                        self.save_drawing()
                    elif event.key == pygame.K_l:
                        self.load_drawing()
                    elif event.key == pygame.K_UP:
                        self.increase_threshold()
                    elif event.key == pygame.K_DOWN:
                        self.decrease_threshold()

            self.draw()
            clock.tick(60)

    def start_drawing(self, pos: Tuple[int, int]) -> None:
        """Start a new drawing path."""
        self.current_path = []
        self.is_drawing = True
        self.add_point(pos)

    def continue_drawing(self, pos: Tuple[int, int]) -> None:
        """Add a point while drawing."""
        self.add_point(pos)

    def finish_drawing(self) -> None:
        """Finish drawing and trigger recognition."""
        self.is_drawing = False
        self.recognize_shape()

    def add_point(self, pos: Tuple[int, int]) -> None:
        """Add a point to the current path."""
        x, y = pos
        point = {
            "x": float(x),
            "y": float(y),
            "t": float(pygame.time.get_ticks()),
        }
        self.current_path.append(point)

    def recognize_shape(self) -> None:
        """Recognize the drawn shape using the $1 recognizer."""
        if len(self.current_path) < 3:
            self.recognition_result = "Draw more points!"
            self.similarity_score = 0.0
            return
        try:
            result, score = classify_path_with_score(self.current_path)
            self.recognition_result = result
            self.similarity_score = score
        except Exception as e:  # pragma: no cover
            self.recognition_result = f"Error: {e}"
            self.similarity_score = 0.0

    def clear_screen(self) -> None:
        """Clear the drawing and results."""
        self.current_path = []
        self.recognition_result = None
        self.similarity_score = 0.0

    def save_drawing(self) -> None:
        """Save the current drawing to a JSON file."""
        if not self.current_path:
            return
        data = {
            "path": self.current_path,
            "recognized_as": self.recognition_result,
            "similarity": self.similarity_score,
        }
        with open("saved_drawing.json", "w") as f:
            json.dump(data, f, indent=2)

    def load_drawing(self) -> None:
        """Load a previously saved drawing."""
        try:
            with open("saved_drawing.json", "r") as f:
                data = json.load(f)
            self.current_path = data["path"]
            self.recognition_result = data["recognized_as"]
            self.similarity_score = data["similarity"]
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
            "Draw shapes in the gray box below:",
            "• Rectangle: Draw a 4‑sided box",
            "• Circle: Draw a round shape",
            "• Triangle: Draw a 3‑sided shape",
            "",
            "Controls:",
            "C: Clear   R: Recognize   S: Save   L: Load",
            "UP/DOWN: Adjust threshold",
            f"Current threshold: {current_threshold:.2f}",
            "Draw SLOWLY and CLEARLY for best results!",
        ]
        y = 10
        for line in instructions:
            txt = self.small_font.render(line, True, self.BLACK)
            self.screen.blit(txt, (10, y))
            y += 30
        # Area label
        label = self.font.render("DRAWING AREA", True, self.GRAY)
        self.screen.blit(label, (400, 160))
        # Available shapes
        shape_text = "Available shapes: " + ", ".join(self.shapes)
        self.screen.blit(
            self.small_font.render(shape_text, True, self.GRAY), (10, 200)
        )
        # Draw current path
        if self.current_path:
            pts = [
                (p["x"], p["y"]) for p in self.current_path
            ]
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
            self.screen.blit(
                self.font.render(result_text, True, self.GREEN), (1200, 1300)
            )
            self.screen.blit(
                self.font.render(score_text, True, self.GREEN), (1200, 1350)
            )
        pygame.display.flip()


def main() -> None:
    """Entry point for the demo."""
    demo = ShapeRecognitionDemo()
    try:
        demo.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
