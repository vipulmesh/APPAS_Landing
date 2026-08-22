#!/usr/bin/env python3
"""Run the experimental webcam-based e-ArUco perception display."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aruco.embedded_app import run


if __name__ == "__main__":
    run()
