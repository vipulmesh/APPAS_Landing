"""Pure, testable image-space localization helpers for ArUco markers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment classification based solely on image-space pixel errors."""

    error_x: float
    error_y: float
    status: str
    instructions: tuple[str, ...]
    aligned: bool


def marker_center(corners: Sequence[Sequence[float]] | np.ndarray) -> Point:
    """Return the average of four marker corners, rounded to pixel coordinates."""
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    if len(points) != 4:
        raise ValueError("ArUco marker corners must contain exactly four points.")
    center = points.mean(axis=0)
    return int(round(center[0])), int(round(center[1]))


def frame_center(frame_width: int, frame_height: int) -> Point:
    """Return the center of a frame without assuming a fixed resolution."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame width and height must be positive.")
    return frame_width // 2, frame_height // 2


def alignment_error(marker: Point, image_center: Point) -> tuple[float, float]:
    """Return marker-minus-frame-center image-space pixel error."""
    return marker[0] - image_center[0], marker[1] - image_center[1]


def classify_alignment(
    error_x: float, error_y: float, x_tolerance: float, y_tolerance: float
) -> AlignmentResult:
    """Classify marker position; positive X is right and positive Y is down."""
    if x_tolerance < 0 or y_tolerance < 0:
        raise ValueError("Alignment tolerances must be non-negative.")

    directions: list[str] = []
    instructions: list[str] = []
    if error_x > x_tolerance:
        directions.append("RIGHT")
        instructions.append("MOVE RIGHT")
    elif error_x < -x_tolerance:
        directions.append("LEFT")
        instructions.append("MOVE LEFT")
    if error_y > y_tolerance:
        directions.append("DOWN")
        instructions.append("MOVE DOWN")
    elif error_y < -y_tolerance:
        directions.append("UP")
        instructions.append("MOVE UP")

    aligned = not directions
    return AlignmentResult(
        error_x=error_x,
        error_y=error_y,
        status="ALIGNED" if aligned else " / ".join(directions),
        instructions=tuple(instructions),
        aligned=aligned,
    )
