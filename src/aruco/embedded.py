"""Experimental e-ArUco construction and interpretation utilities.

An e-ArUco replaces the *central black encoding cell* of a 7x7 outer marker
with a complete smaller 7x7 marker.  It is deliberately not an image overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import ArucoDetector, DetectedMarker
from .localization import marker_center


@dataclass(frozen=True)
class EmbeddedMarkerDiagnostics:
    outer_id: int
    inner_id: int
    outer_center_is_black: bool
    inner_black_ratio: float
    valid: bool


@dataclass(frozen=True)
class EmbeddedDetection:
    """Selected e-ArUco detection state; no flight-control semantics are implied."""

    outer: DetectedMarker | None
    inner: DetectedMarker | None
    active: DetectedMarker | None
    active_kind: str | None
    range_state: str


def _dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install opencv-contrib-python.")
    if not dictionary_name.startswith("DICT_7X7_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError("Embedded ArUco requires a valid OpenCV DICT_7X7_* dictionary.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _marker_image(dictionary, marker_id: int, side_pixels: int) -> np.ndarray:
    if marker_id < 0 or marker_id >= dictionary.bytesList.shape[0]:
        raise ValueError(f"Marker ID {marker_id} is invalid for the selected dictionary.")
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels, borderBits=1)
    image = np.zeros((side_pixels, side_pixels), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, side_pixels, image, 1)
    return image


def _encoding_cells(dictionary, marker_id: int) -> np.ndarray:
    """Return the 7x7 payload cells as booleans where True means black."""
    marker_size = int(dictionary.markerSize)
    total_modules = marker_size + 2
    image = _marker_image(dictionary, marker_id, total_modules * 8)
    cells = image.reshape(total_modules, 8, total_modules, 8).mean(axis=(1, 3)) < 128
    return cells[1:-1, 1:-1]


def outer_center_is_black(dictionary_name: str, outer_id: int) -> bool:
    """Check whether the central outer encoding cell can be replaced safely."""
    dictionary = _dictionary(dictionary_name)
    cells = _encoding_cells(dictionary, outer_id)
    return bool(cells[cells.shape[0] // 2, cells.shape[1] // 2])


def inner_black_cell_ratio(dictionary_name: str, inner_id: int) -> float:
    """Return black payload-cell count divided by total payload cells."""
    dictionary = _dictionary(dictionary_name)
    return float(_encoding_cells(dictionary, inner_id).mean())


def find_valid_outer_ids(dictionary_name: str) -> list[int]:
    """Return IDs with a black central encoding cell, suitable for embedding."""
    dictionary = _dictionary(dictionary_name)
    return [marker_id for marker_id in range(dictionary.bytesList.shape[0]) if outer_center_is_black(dictionary_name, marker_id)]


def validate_embedded_configuration(
    dictionary_name: str, outer_id: int, inner_id: int, min_inner_black_ratio: float = 0.5
) -> EmbeddedMarkerDiagnostics:
    """Validate IDs and paper-inspired cell/black-ratio constraints with diagnostics."""
    if outer_id == inner_id:
        raise ValueError("Outer and inner e-ArUco IDs must be different.")
    if not 0.0 <= min_inner_black_ratio <= 1.0:
        raise ValueError("Minimum inner black-cell ratio must be between 0 and 1.")
    outer_black = outer_center_is_black(dictionary_name, outer_id)
    inner_ratio = inner_black_cell_ratio(dictionary_name, inner_id)
    return EmbeddedMarkerDiagnostics(
        outer_id, inner_id, outer_black, inner_ratio, outer_black and inner_ratio >= min_inner_black_ratio
    )


def validate_physical_dimensions(outer_size_mm: float, inner_size_mm: float, total_modules: int = 9) -> float:
    """Validate square print dimensions against the one-central-cell geometry."""
    if outer_size_mm <= 0 or inner_size_mm <= 0:
        raise ValueError("Outer and inner physical sizes must be positive.")
    ratio = inner_size_mm / outer_size_mm
    required_ratio = 1.0 / total_modules
    if not np.isclose(ratio, required_ratio):
        raise ValueError(f"Inner/outer physical size ratio must be {required_ratio:.6f}; received {ratio:.6f}.")
    return ratio


def generate_embedded_marker(
    dictionary_name: str,
    outer_id: int,
    inner_id: int,
    output_size: int,
    output_path: str | Path,
    min_inner_black_ratio: float = 0.5,
    margin: int = 0,
    inner_ratio: float | None = None,
    outer_size_mm: float | None = None,
    inner_size_mm: float | None = None,
) -> tuple[Path, EmbeddedMarkerDiagnostics]:
    """Generate one geometry-aligned e-ArUco PNG and its validation diagnostics."""
    dictionary = _dictionary(dictionary_name)
    marker_size = int(dictionary.markerSize)
    total_modules = marker_size + 2
    required_ratio = 1.0 / total_modules
    if inner_ratio is not None and not np.isclose(inner_ratio, required_ratio):
        raise ValueError(
            f"A complete inner {marker_size}x{marker_size} marker must occupy one outer cell: "
            f"inner ratio must be {required_ratio:.6f}."
        )
    if (outer_size_mm is None) != (inner_size_mm is None):
        raise ValueError("Provide both outer and inner physical sizes, or neither.")
    if outer_size_mm is not None:
        validate_physical_dimensions(outer_size_mm, inner_size_mm, total_modules)
    if output_size <= 2 * margin or margin < 0:
        raise ValueError("Output size must exceed twice the non-negative margin.")
    diagnostics = validate_embedded_configuration(dictionary_name, outer_id, inner_id, min_inner_black_ratio)
    if not diagnostics.outer_center_is_black:
        valid_ids = find_valid_outer_ids(dictionary_name)
        raise ValueError(f"Outer ID {outer_id} has a white central encoding cell. Try one of: {valid_ids[:10]}")
    if diagnostics.inner_black_ratio < min_inner_black_ratio:
        raise ValueError(
            f"Inner ID {inner_id} black-cell ratio is {diagnostics.inner_black_ratio:.2f}, below {min_inner_black_ratio:.2f}."
        )

    # Work in an exact module lattice: each outer cell is a full inner marker.
    lattice_side = total_modules * total_modules
    outer = _marker_image(dictionary, outer_id, lattice_side)
    inner = _marker_image(dictionary, inner_id, total_modules)
    center_cell = total_modules // 2
    start = center_cell * total_modules
    outer[start : start + total_modules, start : start + total_modules] = inner

    marker_side = output_size - 2 * margin
    rendered = cv2.resize(outer, (marker_side, marker_side), interpolation=cv2.INTER_NEAREST)
    image = np.full((output_size, output_size), 255, dtype=np.uint8)
    image[margin : margin + marker_side, margin : margin + marker_side] = rendered
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write embedded marker image to {path}")
    return path, diagnostics


def interpret_embedded_markers(
    markers: list[DetectedMarker], outer_id: int, inner_id: int, both_visible_policy: str = "outer"
) -> EmbeddedDetection:
    """Choose an active marker for perception according to the configured policy."""
    if both_visible_policy not in {"outer", "inner"}:
        raise ValueError("Both-visible policy must be 'outer' or 'inner'.")
    outer = next((marker for marker in markers if marker.marker_id == outer_id), None)
    inner = next((marker for marker in markers if marker.marker_id == inner_id), None)
    if outer and inner:
        active_kind = both_visible_policy.upper()
        return EmbeddedDetection(outer, inner, outer if both_visible_policy == "outer" else inner, active_kind, "BOTH VISIBLE")
    if outer:
        return EmbeddedDetection(outer, None, outer, "OUTER", "LONG/MEDIUM RANGE")
    if inner:
        return EmbeddedDetection(None, inner, inner, "INNER", "CLOSE RANGE")
    return EmbeddedDetection(None, None, None, None, "NO LANDING MARKER")


def detect_embedded_markers(
    frame: np.ndarray, detector: ArucoDetector, outer_id: int, inner_id: int
) -> list[DetectedMarker]:
    """Detect the outer marker, then rectify its central cell to inspect the inner.

    The same normal ``ArucoDetector`` is reused.  The crop supplies the white
    quiet zone that an inner marker embedded in a black outer cell cannot have
    in the complete image, while retaining the outer's unmodified black bit.
    """
    markers = detector.detect(frame)
    outer = detector.select_target(markers, outer_id)
    if outer is None or detector.select_target(markers, inner_id) is not None:
        return markers
    total_modules = int(detector.dictionary.markerSize) + 2
    canonical_side = total_modules * 100
    destination = np.float32([
        [0, 0], [canonical_side - 1, 0], [canonical_side - 1, canonical_side - 1], [0, canonical_side - 1]
    ])
    transform = cv2.getPerspectiveTransform(outer.corners.astype(np.float32), destination)
    rectified = cv2.warpPerspective(frame, transform, (canonical_side, canonical_side), borderValue=(255, 255, 255))
    cell_side = canonical_side // total_modules
    start = (total_modules // 2) * cell_side
    central_cell = rectified[start : start + cell_side, start : start + cell_side]
    padding = cell_side // 2
    inner_input = cv2.copyMakeBorder(central_cell, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    inner = detector.select_target(detector.detect(inner_input), inner_id)
    if inner is None:
        return markers
    canonical_corners = inner.corners - np.array([padding - start, padding - start], dtype=np.float32)
    inverse_transform = cv2.getPerspectiveTransform(destination, outer.corners.astype(np.float32))
    frame_corners = cv2.perspectiveTransform(canonical_corners.reshape(1, 4, 2).astype(np.float32), inverse_transform)[0]
    embedded_inner = DetectedMarker(inner_id, frame_corners, marker_center(frame_corners))
    return [*markers, embedded_inner]
