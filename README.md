# Landing Marker Detection

Real-time landing-marker detection using classical computer vision and OpenCV.

## What it does

- Opens the webcam
- Reads frames continuously
- Converts frames to grayscale
- Applies Gaussian blur
- Runs Canny edge detection or thresholding
- Detects contours
- Filters contours by area
- Draws contours and bounding boxes
- Displays the live annotated output
- Shows FPS

## Project Structure

```text
Landing-Marker-Detection/
├── images/
├── markers/
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── detection.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
├── README.md
└── project_info.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## ArUco landing-marker perception

The original contour-based detector remains available through `src/main.py`.
The ArUco stage is a separate, frame-based perception pipeline for a landing
marker. It uses OpenCV's native ArUco implementation to detect every visible
marker, then uses only the configured target ID for alignment.

### Configure the target

Edit the ArUco values in `src/config.py`:

```python
ARUCO_DICTIONARY = "DICT_4X4_50"
TARGET_MARKER_ID = 23
ARUCO_X_TOLERANCE = 20
ARUCO_Y_TOLERANCE = 20
```

Tolerances and alignment errors are **image-space pixels**, not metres or a
physical displacement. Positive X means right of the frame centre; positive Y
means below it. `RIGHT / DOWN`, for example, means the target centre is to the
right and below the frame centre. The application is perception-only and never
commands a UAV.

### Generate a marker

Install dependencies (the contrib OpenCV build provides `cv2.aruco`), then
generate the default target marker:

```bash
python scripts/generate_marker.py
```

Or choose an ID and image size:

```bash
python scripts/generate_marker.py --id 23 --size 600
```

The PNG is written to `markers/generated/marker_<id>.png`, with a configurable
white outer margin to aid detection. Print it without cropping or display it at
its native square aspect ratio for webcam testing.
It can later be used as a Gazebo texture; Gazebo does not generate the marker.

### Run the ArUco webcam detector

```bash
python scripts/run_aruco_detector.py
```

Press `q` or `Esc` to exit. The live window shows all marker outlines and IDs,
and highlights the configured target with its centre, frame centre, alignment
vector, pixel X/Y errors, status, instructions, and FPS. When the configured
target is absent it continues running and displays `TARGET MARKER NOT DETECTED`.

### Test

```bash
pytest
```

The non-camera tests cover centre/frame calculations, pixel errors, alignment
directions and tolerance boundaries, and target-ID selection.

### Future camera interface

`ArucoDetector.detect(frame)` accepts a numpy image and has no webcam
dependency. Today the application supplies frames from `cv2.VideoCapture`;
later a Gazebo camera can supply its numpy frame to the same detector without
duplicating perception logic.

Not implemented yet: calibration, pose estimation (`solvePnP`), physical
distance estimation, Gazebo integration, UAV control, autonomous landing, and
e-ArUco.
