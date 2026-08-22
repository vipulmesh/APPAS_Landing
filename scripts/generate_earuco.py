#!/usr/bin/env python3
"""Generate and validate one geometry-aligned experimental e-ArUco PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aruco.embedded import detect_embedded_markers, generate_embedded_marker, generate_standalone_inner_marker, marker_quality
from src.aruco.detector import ArucoDetector
from src.config import (
    EARUCO_DICTIONARY, EARUCO_INNER_ID, EARUCO_INNER_QUIET_RATIO, EARUCO_INNER_RATIO,
    EARUCO_MARGIN, EARUCO_MIN_INNER_BLACK_RATIO, EARUCO_MIN_MODULE_PIXELS, EARUCO_OUTPUT_SIZE,
    EARUCO_OUTER_ID, EARUCO_OUTER_SIZE_MM, EARUCO_INNER_SIZE_MM,
)


def _write_detection_debug_image(image, markers, output: Path) -> None:
    annotated = image.copy()
    for marker in markers:
        corners = marker.corners.astype("int32").reshape(-1, 1, 2)
        cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
        cv2.circle(annotated, marker.center, 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"ID {marker.marker_id}", tuple(corners[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 0, 0), 2)
    cv2.imwrite(str(output), annotated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an experimental embedded 7x7 ArUco marker.")
    parser.add_argument("--dictionary", default=EARUCO_DICTIONARY)
    parser.add_argument("--outer-id", type=int, default=EARUCO_OUTER_ID)
    parser.add_argument("--inner-id", type=int, default=EARUCO_INNER_ID)
    parser.add_argument("--size", type=int, default=EARUCO_OUTPUT_SIZE)
    parser.add_argument("--margin", type=int, default=EARUCO_MARGIN)
    parser.add_argument("--min-inner-black-ratio", type=float, default=EARUCO_MIN_INNER_BLACK_RATIO)
    parser.add_argument("--inner-ratio", type=float, default=EARUCO_INNER_RATIO)
    parser.add_argument("--quiet-ratio", type=float, default=EARUCO_INNER_QUIET_RATIO)
    parser.add_argument("--min-module-pixels", type=int, default=EARUCO_MIN_MODULE_PIXELS)
    parser.add_argument("--outer-size-mm", type=float, default=EARUCO_OUTER_SIZE_MM)
    parser.add_argument("--inner-size-mm", type=float, default=EARUCO_INNER_SIZE_MM)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "markers" / "generated" / f"earuco_outer{args.outer_id}_inner{args.inner_id}.png"
    try:
        path, report = generate_embedded_marker(
            args.dictionary, args.outer_id, args.inner_id, args.size, output,
            args.min_inner_black_ratio, args.margin, args.inner_ratio, args.quiet_ratio,
            args.min_module_pixels, args.outer_size_mm, args.inner_size_mm,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Generated e-ArUco: {path}")
    print(f"Outer ID: {report.outer_id}; center cell: {'BLACK' if report.outer_center_is_black else 'WHITE'}")
    print(f"Inner ID: {report.inner_id}; black-cell ratio: {report.inner_black_ratio:.2f}")
    print(f"Physical layout: {args.outer_size_mm:g} mm outer / {args.inner_size_mm:g} mm inner")
    standalone = ROOT / "markers" / "generated" / "earuco_inner_debug.png"
    generate_standalone_inner_marker(args.dictionary, args.inner_id, args.size, standalone, args.inner_ratio, args.min_module_pixels)
    direct_detector = ArucoDetector(args.dictionary)
    image = cv2.imread(str(path))
    direct = direct_detector.detect(image)
    detected = detect_embedded_markers(image, direct_detector, args.outer_id, args.inner_id, args.inner_ratio, args.quiet_ratio, args.min_module_pixels)
    print("Standalone inner IDs:", [item.marker_id for item in direct_detector.detect(cv2.imread(str(standalone)))])
    print("Final-image direct IDs:", [item.marker_id for item in direct])
    print("Specialized e-ArUco IDs:", [(item.marker_id, item.center, marker_quality(item).module_pixels) for item in detected])
    debug = ROOT / "markers" / "generated" / "earuco_debug.png"
    _write_detection_debug_image(image, detected, debug)
    print(f"Detection debug image: {debug}")


if __name__ == "__main__":
    main()
