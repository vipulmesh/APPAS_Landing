"""Experimental e-ArUco rendering and specialized outer-marker decoding."""

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


@dataclass(frozen=True)
class MarkerQuality:
    area: float
    perimeter: float
    module_pixels: float
    geometry_valid: bool


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


def validate_physical_dimensions(outer_size_mm: float, inner_size_mm: float, expected_ratio: float) -> float:
    """Validate positive print dimensions against the configured marker ratio."""
    if outer_size_mm <= 0 or inner_size_mm <= 0:
        raise ValueError("Outer and inner physical sizes must be positive.")
    ratio = inner_size_mm / outer_size_mm
    if not np.isclose(ratio, expected_ratio):
        raise ValueError(f"Inner/outer physical size ratio must be {expected_ratio:.6f}; received {ratio:.6f}.")
    return ratio


def _layout(marker_side: int, total_modules: int, inner_ratio: float, quiet_ratio: float, min_module_pixels: int) -> tuple[int, int, int]:
    if not 0.10 <= inner_ratio <= 0.30 or quiet_ratio < 0:
        raise ValueError("Inner ratio must be 0.10–0.30 and quiet-zone ratio must be non-negative.")
    inner_side = int(marker_side * inner_ratio) // total_modules * total_modules
    quiet = int(round(marker_side * quiet_ratio))
    if inner_side < total_modules * min_module_pixels:
        raise ValueError(f"Inner module size is below the required {min_module_pixels} pixels.")
    if inner_side + (2 * quiet) >= marker_side:
        raise ValueError("Inner marker and quiet zone do not fit inside the outer marker.")
    return inner_side, quiet, (marker_side - inner_side) // 2


def generate_embedded_marker(
    dictionary_name: str,
    outer_id: int,
    inner_id: int,
    output_size: int,
    output_path: str | Path,
    min_inner_black_ratio: float = 0.5,
    margin: int = 0,
    inner_ratio: float = 0.20,
    quiet_ratio: float = 1.0 / 60.0,
    min_module_pixels: int = 10,
    outer_size_mm: float | None = None,
    inner_size_mm: float | None = None,
) -> tuple[Path, EmbeddedMarkerDiagnostics]:
    """Generate one geometry-aligned e-ArUco PNG and its validation diagnostics."""
    dictionary = _dictionary(dictionary_name)
    marker_size = int(dictionary.markerSize)
    total_modules = marker_size + 2
    if (outer_size_mm is None) != (inner_size_mm is None):
        raise ValueError("Provide both outer and inner physical sizes, or neither.")
    if outer_size_mm is not None:
        validate_physical_dimensions(outer_size_mm, inner_size_mm, inner_ratio)
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

    marker_side = output_size - 2 * margin
    inner_side, quiet, start = _layout(marker_side, total_modules, inner_ratio, quiet_ratio, min_module_pixels)
    # Both markers are natively rendered at their final module resolution. The
    # white quiet zone is part of the final image, so the inner is directly
    # detectable without ROI padding or fabricated coordinates.
    rendered = _marker_image(dictionary, outer_id, marker_side)
    rendered[start - quiet : start + inner_side + quiet, start - quiet : start + inner_side + quiet] = 255
    rendered[start : start + inner_side, start : start + inner_side] = _marker_image(dictionary, inner_id, inner_side)
    image = np.full((output_size, output_size), 255, dtype=np.uint8)
    image[margin : margin + marker_side, margin : margin + marker_side] = rendered
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write embedded marker image to {path}")
    return path, diagnostics


def generate_standalone_inner_marker(
    dictionary_name: str, inner_id: int, output_size: int, output_path: str | Path,
    inner_ratio: float = 0.20, min_module_pixels: int = 10,
) -> Path:
    """Generate a centered inner marker with a real white quiet zone for diagnostics."""
    dictionary = _dictionary(dictionary_name)
    total_modules = int(dictionary.markerSize) + 2
    inner_side, _, start = _layout(output_size, total_modules, inner_ratio, 0, min_module_pixels)
    image = np.full((output_size, output_size), 255, dtype=np.uint8)
    image[start : start + inner_side, start : start + inner_side] = _marker_image(dictionary, inner_id, inner_side)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write standalone inner marker image to {path}")
    return path


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
    frame: np.ndarray, detector: ArucoDetector, outer_id: int, inner_id: int,
    inner_ratio: float = 0.20, quiet_ratio: float = 1.0 / 60.0, min_module_pixels: int = 10,
) -> list[DetectedMarker]:
    """Detect the inner directly and recover outer candidates with a known mask.

    The inner ID is accepted only if the unmodified full-frame OpenCV detector
    finds it. The specialized outer decoder uses rejected quadrilaterals,
    rectifies actual image pixels, restores the intentionally embedded central
    region from the configured outer code, then re-runs the same detector.
    """
    markers = detector.detect(frame)
    if detector.select_target(markers, outer_id) is not None:
        return markers
    total_modules = int(detector.dictionary.markerSize) + 2
    canonical_side = total_modules * 120
    inner_side, quiet, start = _layout(canonical_side, total_modules, inner_ratio, quiet_ratio, min_module_pixels)
    reference = _marker_image(detector.dictionary, outer_id, canonical_side)
    destination = np.float32([
        [0, 0], [canonical_side - 1, 0], [canonical_side - 1, canonical_side - 1], [0, canonical_side - 1]
    ])
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    for candidate in sorted(detector.rejected_candidates(frame), key=lambda points: abs(cv2.contourArea(points.astype(np.float32))), reverse=True):
        if abs(cv2.contourArea(candidate.astype(np.float32))) < 400:
            continue
        transform = cv2.getPerspectiveTransform(candidate.astype(np.float32), destination)
        rectified = cv2.warpPerspective(grayscale, transform, (canonical_side, canonical_side), flags=cv2.INTER_NEAREST, borderValue=255)
        restored = rectified.copy()
        restored[start - quiet : start + inner_side + quiet, start - quiet : start + inner_side + quiet] = reference[start - quiet : start + inner_side + quiet, start - quiet : start + inner_side + quiet]
        padding = max(20, canonical_side // 18)
        recovered = detector.select_target(detector.detect(cv2.copyMakeBorder(restored, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))), outer_id)
        if recovered is None:
            continue
        canonical_corners = recovered.corners - padding
        inverse_transform = cv2.getPerspectiveTransform(destination, candidate.astype(np.float32))
        frame_corners = cv2.perspectiveTransform(canonical_corners.reshape(1, 4, 2).astype(np.float32), inverse_transform)[0]
        return [*markers, DetectedMarker(outer_id, frame_corners, marker_center(frame_corners))]
    return markers


def marker_quality(marker: DetectedMarker, module_count: int = 9) -> MarkerQuality:
    """Return geometric diagnostics derived from real detected corners."""
    corners = marker.corners.astype(np.float32)
    area = abs(float(cv2.contourArea(corners)))
    perimeter = float(cv2.arcLength(corners, True))
    edges = [float(np.linalg.norm(corners[(index + 1) % 4] - corners[index])) for index in range(4)]
    aspect = max(edges) / min(edges) if min(edges) else float("inf")
    return MarkerQuality(area, perimeter, min(edges) / module_count, area > 0 and aspect <= 1.4)
