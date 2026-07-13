import cv2

try:
    from config import CONTOUR_APPROX_EPSILON_RATIO, MAX_CONTOUR_AREA_RATIO, MIN_CONTOUR_AREA
except ImportError:
    from .config import CONTOUR_APPROX_EPSILON_RATIO, MAX_CONTOUR_AREA_RATIO, MIN_CONTOUR_AREA


def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def detect_landing_marker(frame, binary_frame):
    frame_area = frame.shape[0] * frame.shape[1]
    max_allowed_area = frame_area * MAX_CONTOUR_AREA_RATIO

    contour_result = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > max_allowed_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, CONTOUR_APPROX_EPSILON_RATIO * perimeter, True)
        bbox = cv2.boundingRect(contour)
        center = get_contour_center(contour)
        candidates.append((contour, bbox, center, area, len(approx)))

    if not candidates:
        return None, contours

    candidates.sort(key=lambda item: (item[4], -item[3]))

    for candidate in candidates:
        if candidate[4] >= 4:
            return candidate[:4], contours

    best_candidate = max(candidates, key=lambda item: item[3])
    return best_candidate[:4], contours
