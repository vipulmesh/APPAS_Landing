# APASS – Autonomous Landing Marker Detection using Python & OpenCV

## Project Information

### Project Title
**Autonomous Landing Marker Detection using Python and OpenCV**

---

## Problem Statement

APASS requires an autonomous drone to recognize a designated landing station and land accurately without human intervention. The objective of this project is to develop a basic computer vision application using **Python** and **OpenCV** that detects a landing marker in real time and provides positional information for precise drone alignment and autonomous landing.

---

## Project Objective

The goal of this project is to build a real-time computer vision system capable of:

- Detecting a predefined landing marker from a live camera feed.
- Identifying the marker's position within the image.
- Calculating the center coordinates (X, Y) of the detected marker.
- Providing visual feedback for alignment.
- Serving as the perception module for future autonomous drone landing systems.

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.x | Programming Language |
| OpenCV | Computer Vision & Image Processing |
| NumPy | Numerical Operations |
| Webcam | Live Video Input |
| VS Code / PyCharm | Development Environment |

---

## Features

- Live webcam video capture
- Real-time landing marker detection
- Bounding box visualization
- Center point detection
- Display X-Y coordinates
- Continuous frame processing
- Modular and beginner-friendly implementation

---

## System Workflow

```
                  +----------------+
                  |   Webcam Feed  |
                  +--------+-------+
                           |
                           v
                  +----------------+
                  | Capture Frames |
                  +--------+-------+
                           |
                           v
                  +----------------------+
                  | Image Preprocessing  |
                  | Grayscale + Blur     |
                  +--------+-------------+
                           |
                           v
                  +----------------------+
                  | Threshold / Edge     |
                  | Detection            |
                  +--------+-------------+
                           |
                           v
                  +----------------------+
                  | Contour Detection    |
                  +--------+-------------+
                           |
                           v
                  +----------------------+
                  | Marker Recognition   |
                  +--------+-------------+
                           |
                           v
                  +----------------------+
                  | Center Calculation   |
                  +--------+-------------+
                           |
                           v
                  +----------------------+
                  | Display Result       |
                  | Bounding Box + X,Y   |
                  +----------------------+
```

---

## Input

- Live webcam feed
- Printed landing marker

---

## Output

- Detected landing marker
- Bounding rectangle
- Center point
- X-Y coordinates
- Detection status
- Processed live video

---

## Project Modules

### Module 1 – Video Capture
- Capture frames from webcam.
- Handle continuous video stream.

### Module 2 – Image Preprocessing
- Convert RGB image to grayscale.
- Apply Gaussian Blur.
- Reduce image noise.

### Module 3 – Marker Detection
- Apply thresholding or edge detection.
- Detect contours.
- Filter candidate shapes.
- Identify landing marker.

### Module 4 – Position Estimation
- Calculate contour centroid.
- Determine X-Y coordinates.
- Estimate marker location.

### Module 5 – Visualization
- Draw contour.
- Draw bounding box.
- Mark center point.
- Display coordinates.
- Show detection status.

---

## Folder Structure

```
Landing-Marker-Detection/
│
├── images/
├── markers/
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── detection.py
│   ├── utils.py
│   └── config.py
│
├── requirements.txt
├── README.md
└── project_info.md
```

---

## Functional Requirements

- Capture video from webcam.
- Detect the landing marker.
- Highlight the detected marker.
- Compute marker center.
- Display X-Y coordinates.
- Process frames in real time.

---

## Non-Functional Requirements

- Real-time performance (15–30 FPS recommended)
- Easy to understand and modify
- Modular code structure
- Cross-platform compatibility
- Low computational requirements

---

## Expected Outcome

The application should successfully detect a landing marker from a live webcam feed, highlight it with a bounding box, calculate its center coordinates, and display the processed output in real time. This prototype demonstrates the core computer vision component required for autonomous drone landing.

---

## Future Enhancements

- ArUco Marker Detection
- AprilTag Detection
- Distance Estimation
- Pose Estimation (Yaw, Pitch, Roll)
- Camera Calibration
- Multi-Marker Detection
- Autonomous Drone Landing
- PX4 Integration
- ArduPilot Integration
- ROS2 Support
- AI-based Marker Detection using YOLO

---

## Applications

- Autonomous Drone Landing
- UAV Navigation
- Drone Delivery Systems
- Search and Rescue Missions
- Precision Agriculture
- Robotics
- Warehouse Automation
- Industrial Inspection

---

## Learning Outcomes

After completing this project, you will understand:

- Python programming
- OpenCV fundamentals
- Image preprocessing
- Edge detection
- Contour detection
- Shape recognition
- Coordinate calculation
- Real-time computer vision
- Modular project development

---

## Project Status

**Version:** 1.0 (Prototype)

**Current Scope:** Computer Vision-based Landing Marker Detection

**Future Scope:** Integration with drone flight controllers (PX4, ArduPilot) for fully autonomous landing.