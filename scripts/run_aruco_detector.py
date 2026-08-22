#!/usr/bin/env python3
"""Run the webcam-based ArUco landing-marker detector."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aruco.app import run


if __name__ == "__main__":
    run()
