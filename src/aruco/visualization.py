"""Drawing layer for the ArUco alignment application."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import DetectedMarker
from .localization import AlignmentResult, Point


def draw_aruco_visualization(
    frame: np.ndarray,
    markers: list[DetectedMarker],
    target: DetectedMarker | None,
    image_center: Point,
    alignment: AlignmentResult | None,
    target_marker_id: int,
    fps: float,
) -> np.ndarray:
    """Return an annotated frame; only the target contributes to alignment."""
    output = frame.copy()
    for marker in markers:
        points = marker.corners.astype(np.int32).reshape((-1, 1, 2))
        is_target = marker is target
        color = (0, 255, 0) if is_target else (120, 120, 120)
        cv2.polylines(output, [points], True, color, 2)
        cv2.putText(output, f"ID: {marker.marker_id}", tuple(points[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.drawMarker(output, image_center, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    lines = [f"Target ID: {target_marker_id}", f"FPS: {fps:.1f}"]
    if target is None or alignment is None:
        lines.extend(["TARGET MARKER NOT DETECTED", f"Frame Center: {image_center}"])
        text_color = (0, 165, 255)
    else:
        cv2.circle(output, target.center, 6, (0, 0, 255), -1)
        cv2.line(output, image_center, target.center, (255, 255, 0), 2)
        lines.extend([
            f"Marker Center: {target.center}",
            f"Frame Center: {image_center}",
            f"Error X: {alignment.error_x:+.0f}px",
            f"Error Y: {alignment.error_y:+.0f}px",
            f"Status: {alignment.status}",
            f"Instruction: {' / '.join(alignment.instructions) if alignment.instructions else 'HOLD POSITION'}",
        ])
        text_color = (0, 255, 0) if alignment.aligned else (255, 255, 255)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (20, 35 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    return output
