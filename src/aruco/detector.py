"""OpenCV-version-isolated ArUco detection operating on numpy frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .localization import Point, marker_center


@dataclass(frozen=True)
class DetectedMarker:
    """An ArUco marker detected in the current camera frame."""

    marker_id: int
    corners: np.ndarray
    center: Point


class ArucoDetector:
    """Reusable ArUco detector; initialize once and call :meth:`detect` per frame."""

    def __init__(self, dictionary_name: str) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV ArUco support is unavailable. Install opencv-contrib-python."
            )
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary_name = dictionary_name
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._modern_detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            parameters = cv2.aruco.DetectorParameters()
            self._modern_detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            self._parameters = cv2.aruco.DetectorParameters_create()
        else:
            raise RuntimeError("Installed OpenCV has an unsupported ArUco detector API.")

    def detect(self, frame: np.ndarray) -> list[DetectedMarker]:
        """Detect all ArUco markers in a BGR or grayscale numpy frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("A non-empty numpy image frame is required.")
        if self._modern_detector is not None:
            corners, ids, _ = self._modern_detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self._parameters)
        if ids is None:
            return []
        return [
            DetectedMarker(int(marker_id), np.asarray(corner).reshape(4, 2), marker_center(corner))
            for marker_id, corner in zip(ids.flatten(), corners)
        ]

    @staticmethod
    def select_target(markers: list[DetectedMarker], target_marker_id: int) -> DetectedMarker | None:
        """Return exactly the configured target marker, never an arbitrary detection."""
        return next((marker for marker in markers if marker.marker_id == target_marker_id), None)
