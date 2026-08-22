"""Frame-based ArUco detection and image-space landing alignment."""

from .detector import ArucoDetector, DetectedMarker
from .embedded import (
    EmbeddedDetection, EmbeddedMarkerDiagnostics, MarkerQuality, detect_embedded_markers,
    generate_embedded_marker, generate_standalone_inner_marker, interpret_embedded_markers, marker_quality,
)
from .localization import AlignmentResult, alignment_error, classify_alignment, frame_center, marker_center

__all__ = [
    "AlignmentResult",
    "ArucoDetector",
    "DetectedMarker",
    "EmbeddedDetection",
    "EmbeddedMarkerDiagnostics",
    "MarkerQuality",
    "detect_embedded_markers",
    "alignment_error",
    "classify_alignment",
    "frame_center",
    "generate_embedded_marker",
    "generate_standalone_inner_marker",
    "interpret_embedded_markers",
    "marker_quality",
    "marker_center",
]
