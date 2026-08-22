#!/usr/bin/env python3
"""Generate a configurable printable ArUco marker PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aruco.generator import generate_marker
from src.config import (
    ARUCO_BORDER_BITS, ARUCO_DICTIONARY, ARUCO_MARKER_MARGIN, ARUCO_MARKER_SIZE, TARGET_MARKER_ID,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an OpenCV ArUco marker PNG.")
    parser.add_argument("--dictionary", default=ARUCO_DICTIONARY, help="OpenCV dictionary name")
    parser.add_argument("--id", type=int, default=TARGET_MARKER_ID, help="Marker ID")
    parser.add_argument("--size", type=int, default=ARUCO_MARKER_SIZE, help="Square marker size in pixels")
    parser.add_argument("--border-bits", type=int, default=ARUCO_BORDER_BITS, help="Black border width in ArUco bits")
    parser.add_argument("--margin", type=int, default=ARUCO_MARKER_MARGIN, help="White outer margin in pixels")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()
    output = args.output or ROOT / "markers" / "generated" / f"marker_{args.id}.png"
    try:
        saved = generate_marker(args.dictionary, args.id, args.size, output, args.border_bits, args.margin)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Generated ArUco marker ID {args.id}: {saved}")


if __name__ == "__main__":
    main()
