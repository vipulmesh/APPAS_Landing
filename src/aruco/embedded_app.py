"""Webcam experiment for e-ArUco outer/inner perception visualization."""

from __future__ import annotations

import time

import cv2

try:
    from config import (
        ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE, CAMERA_INDEX, EARUCO_BOTH_VISIBLE_POLICY,
        EARUCO_DICTIONARY, EARUCO_INNER_ID, EARUCO_OUTER_ID, FRAME_HEIGHT, FRAME_WIDTH,
    )
except ImportError:
    from ..config import (
        ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE, CAMERA_INDEX, EARUCO_BOTH_VISIBLE_POLICY,
        EARUCO_DICTIONARY, EARUCO_INNER_ID, EARUCO_OUTER_ID, FRAME_HEIGHT, FRAME_WIDTH,
    )

from .detector import ArucoDetector
from .embedded import detect_embedded_markers, interpret_embedded_markers
from .embedded_visualization import draw_embedded_visualization
from .localization import alignment_error, classify_alignment, frame_center


def run() -> None:
    """Display e-ArUco interpretation; it remains a perception-only experiment."""
    detector = ArucoDetector(EARUCO_DICTIONARY)
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Check camera permissions and configuration.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    previous_time = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a camera frame; stopping e-ArUco detector.")
                break
            markers = detect_embedded_markers(frame, detector, EARUCO_OUTER_ID, EARUCO_INNER_ID)
            detected = interpret_embedded_markers(markers, EARUCO_OUTER_ID, EARUCO_INNER_ID, EARUCO_BOTH_VISIBLE_POLICY)
            image_center = frame_center(frame.shape[1], frame.shape[0])
            alignment = None
            if detected.active:
                error_x, error_y = alignment_error(detected.active.center, image_center)
                alignment = classify_alignment(error_x, error_y, ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE)
            now = time.perf_counter()
            fps = 1.0 / (now - previous_time) if now > previous_time else 0.0
            previous_time = now
            cv2.imshow("Embedded ArUco Landing Marker Detection", draw_embedded_visualization(frame, detected, image_center, alignment, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
