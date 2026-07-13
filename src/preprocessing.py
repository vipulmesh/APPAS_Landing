import cv2

try:
    from config import (
        CANNY_HIGH_THRESHOLD,
        CANNY_LOW_THRESHOLD,
        GAUSSIAN_BLUR_KERNEL,
        MAX_BINARY_VALUE,
        THRESHOLD_VALUE,
        USE_CANNY,
    )
except ImportError:
    from .config import (
        CANNY_HIGH_THRESHOLD,
        CANNY_LOW_THRESHOLD,
        GAUSSIAN_BLUR_KERNEL,
        MAX_BINARY_VALUE,
        THRESHOLD_VALUE,
        USE_CANNY,
    )


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    if USE_CANNY:
        return cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    _, thresholded = cv2.threshold(
        blurred,
        THRESHOLD_VALUE,
        MAX_BINARY_VALUE,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return thresholded
