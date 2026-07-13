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


