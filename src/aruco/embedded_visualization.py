"""Visualization overlay for e-ArUco outer/inner interpretation."""

from __future__ import annotations

import cv2
import numpy as np

from .embedded import EmbeddedDetection
from .localization import AlignmentResult, Point


def draw_embedded_visualization(
    frame: np.ndarray, detection: EmbeddedDetection, image_center: Point,
    alignment: AlignmentResult | None, fps: float,
) -> np.ndarray:
    """Draw outer/inner boundaries and the currently active perception marker."""
    output = frame.copy()
    for marker, kind, color in ((detection.outer, "OUTER", (255, 0, 0)), (detection.inner, "INNER", (0, 255, 0))):
        if marker is None:
            continue
        points = marker.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, color, 2)
        cv2.putText(output, f"{kind} ID: {marker.marker_id}", tuple(points[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    cv2.drawMarker(output, image_center, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    lines = ["e-ArUco detected" if detection.active else "e-ArUco NOT DETECTED", f"Outer ID: {detection.outer.marker_id if detection.outer else 'NOT DETECTED'}", f"Inner ID: {detection.inner.marker_id if detection.inner else 'NOT DETECTED'}", f"Active marker: {detection.active_kind or 'NONE'}", f"Range: {detection.range_state}", f"FPS: {fps:.1f}"]
    if detection.active is not None and alignment is not None:
        cv2.circle(output, detection.active.center, 6, (0, 0, 255), -1)
        cv2.line(output, image_center, detection.active.center, (255, 255, 0), 2)
        lines.extend([f"Marker Center: {detection.active.center}", f"Error X: {alignment.error_x:+.0f}px", f"Error Y: {alignment.error_y:+.0f}px", f"Status: {alignment.status}"])
    for index, line in enumerate(lines):
        cv2.putText(output, line, (20, 35 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2, cv2.LINE_AA)
    return output
