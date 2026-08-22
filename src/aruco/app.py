"""Webcam entry point for the frame-based ArUco landing-marker pipeline."""

from __future__ import annotations

import time

import cv2

try:
    from config import (
        ARUCO_DICTIONARY, ARUCO_WINDOW_NAME, ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE,
        CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, TARGET_MARKER_ID,
    )
except ImportError:
    from ..config import (
        ARUCO_DICTIONARY, ARUCO_WINDOW_NAME, ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE,
        CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, TARGET_MARKER_ID,
    )

from .detector import ArucoDetector
from .localization import alignment_error, classify_alignment, frame_center
from .visualization import draw_aruco_visualization


def run() -> None:
    """Read webcam frames, detect the configured ID, and display alignment."""
    detector = ArucoDetector(ARUCO_DICTIONARY)
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
                print("Could not read a camera frame; stopping ArUco detector.")
                break
            markers = detector.detect(frame)
            target = detector.select_target(markers, TARGET_MARKER_ID)
            image_center = frame_center(frame.shape[1], frame.shape[0])
            alignment = None
            if target is not None:
                error_x, error_y = alignment_error(target.center, image_center)
                alignment = classify_alignment(error_x, error_y, ARUCO_X_TOLERANCE, ARUCO_Y_TOLERANCE)
            now = time.perf_counter()
            fps = 1.0 / (now - previous_time) if now > previous_time else 0.0
            previous_time = now
            output = draw_aruco_visualization(frame, markers, target, image_center, alignment, TARGET_MARKER_ID, fps)
            cv2.imshow(ARUCO_WINDOW_NAME, output)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
