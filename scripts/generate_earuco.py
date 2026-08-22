#!/usr/bin/env python3
"""Generate and validate one geometry-aligned experimental e-ArUco PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aruco.embedded import detect_embedded_markers, generate_embedded_marker
from src.aruco.detector import ArucoDetector
from src.config import (
    EARUCO_DICTIONARY, EARUCO_INNER_ID, EARUCO_MARGIN, EARUCO_MIN_INNER_BLACK_RATIO,
    EARUCO_OUTPUT_SIZE, EARUCO_OUTER_ID, EARUCO_OUTER_SIZE_MM, EARUCO_INNER_SIZE_MM,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an experimental embedded 7x7 ArUco marker.")
    parser.add_argument("--dictionary", default=EARUCO_DICTIONARY)
    parser.add_argument("--outer-id", type=int, default=EARUCO_OUTER_ID)
    parser.add_argument("--inner-id", type=int, default=EARUCO_INNER_ID)
    parser.add_argument("--size", type=int, default=EARUCO_OUTPUT_SIZE)
    parser.add_argument("--margin", type=int, default=EARUCO_MARGIN)
    parser.add_argument("--min-inner-black-ratio", type=float, default=EARUCO_MIN_INNER_BLACK_RATIO)
    parser.add_argument("--inner-ratio", type=float, help="Must be 1/9 for a 7x7 marker including its border.")
    parser.add_argument("--outer-size-mm", type=float, default=EARUCO_OUTER_SIZE_MM)
    parser.add_argument("--inner-size-mm", type=float, default=EARUCO_INNER_SIZE_MM)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "markers" / "generated" / f"earuco_outer{args.outer_id}_inner{args.inner_id}.png"
    try:
        path, report = generate_embedded_marker(
            args.dictionary, args.outer_id, args.inner_id, args.size, output,
            args.min_inner_black_ratio, args.margin, args.inner_ratio, args.outer_size_mm, args.inner_size_mm,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Generated e-ArUco: {path}")
    print(f"Outer ID: {report.outer_id}; center cell: {'BLACK' if report.outer_center_is_black else 'WHITE'}")
    print(f"Inner ID: {report.inner_id}; black-cell ratio: {report.inner_black_ratio:.2f}")
    print(f"Physical layout: {args.outer_size_mm:g} mm outer / {args.inner_size_mm:g} mm inner")
    detected = detect_embedded_markers(__import__('cv2').imread(str(path)), ArucoDetector(args.dictionary), args.outer_id, args.inner_id)
    print("OpenCV detected:", [(item.marker_id, item.center, item.corners.tolist()) for item in detected])


if __name__ == "__main__":
    main()
