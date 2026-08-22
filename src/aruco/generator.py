"""ArUco marker PNG generation using OpenCV's native ArUco implementation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_marker(
    dictionary_name: str,
    marker_id: int,
    image_size: int,
    output_path: str | Path,
    border_bits: int = 1,
    margin: int = 0,
) -> Path:
    """Generate a printable square marker with an optional white outer margin."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    if image_size <= 0 or border_bits < 1 or margin < 0 or image_size <= 2 * margin:
        raise ValueError("Image size must exceed twice the non-negative margin; border bits must be at least one.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if marker_id < 0 or marker_id >= dictionary.bytesList.shape[0]:
        raise ValueError(f"Marker ID {marker_id} is invalid for {dictionary_name}.")
    marker_size = image_size - (2 * margin)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, borderBits=border_bits)
    elif hasattr(cv2.aruco, "drawMarker"):
        marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker, border_bits)
    else:
        raise RuntimeError("Installed OpenCV has an unsupported ArUco generator API.")
    image = np.full((image_size, image_size), 255, dtype=np.uint8)
    image[margin : margin + marker_size, margin : margin + marker_size] = marker
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write marker image to {path}")
    return path
