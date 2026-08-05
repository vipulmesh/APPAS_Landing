import cv2

try:
    from config import (
        ASPECT_WEIGHT,
        CONFIDENCE_THRESHOLD,
        CONTOUR_APPROX_EPSILON_RATIO,
        CONVEXITY_WEIGHT,
        EXTENT_WEIGHT,
        MAX_ASPECT_RATIO,
        MAX_CONTOUR_AREA_RATIO,
        MIN_ASPECT_RATIO,
        MIN_CONTOUR_AREA,
        MIN_EXTENT,
        MIN_SOLIDITY,
        SOLIDITY_WEIGHT,
        VERTEX_WEIGHT,
    )
except ImportError:
    from .config import (
        ASPECT_WEIGHT,
        CONFIDENCE_THRESHOLD,
        CONTOUR_APPROX_EPSILON_RATIO,
        CONVEXITY_WEIGHT,
        EXTENT_WEIGHT,
        MAX_ASPECT_RATIO,
        MAX_CONTOUR_AREA_RATIO,
        MIN_ASPECT_RATIO,
        MIN_CONTOUR_AREA,
        MIN_EXTENT,
        MIN_SOLIDITY,
        SOLIDITY_WEIGHT,
        VERTEX_WEIGHT,
    )


def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _find_contours(binary_frame):
    contour_result = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contour_result[0] if len(contour_result) == 2 else contour_result[1]


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def _score_from_range(value, lower, upper):
    if value < lower or value > upper:
        return None

    midpoint = (lower + upper) / 2.0
    if midpoint == 0:
        return 1.0

    distance = abs(value - midpoint) / (upper - lower)
    return 1.0 - _clamp(distance)


def _score_candidate(contour, max_allowed_area):
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA or area > max_allowed_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    approx = cv2.approxPolyDP(contour, CONTOUR_APPROX_EPSILON_RATIO * perimeter, True)
    if len(approx) < 4:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return None

    rotated_rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rotated_rect[1]
    if rect_width == 0 or rect_height == 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return None

    polygon_vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)

    aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    extent = area / float(rect_width * rect_height)
    solidity = area / float(hull_area)

    if extent < MIN_EXTENT or solidity < MIN_SOLIDITY:
        return None

    aspect_score = _score_from_range(aspect_ratio, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
    if aspect_score is None:
        return None

    solidity_score = _clamp((solidity - MIN_SOLIDITY) / (1.0 - MIN_SOLIDITY))
    extent_score = _clamp((extent - MIN_EXTENT) / (1.0 - MIN_EXTENT))
    convexity_score = 1.0 if is_convex else 0.0

    if 4 <= polygon_vertices <= 10:
        vertex_score = 1.0
    else:
        vertex_score = _clamp(1.0 - abs(polygon_vertices - 6) / 10.0)

    confidence = (
        ASPECT_WEIGHT * aspect_score
        + SOLIDITY_WEIGHT * solidity_score
        + EXTENT_WEIGHT * extent_score
        + CONVEXITY_WEIGHT * convexity_score
        + VERTEX_WEIGHT * vertex_score
    )

    confidence = _clamp(confidence)
    center = get_contour_center(contour)

    return {
        "contour": contour,
        "bbox": (x, y, w, h),
        "center": center,
        "landing_point": center,
        "area": area,
        "score": confidence,
        "confidence": confidence,
        "vertices": polygon_vertices,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "is_convex": is_convex,
    }


def detect_landing_marker(frame, binary_frame):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    max_allowed_area = frame_area * MAX_CONTOUR_AREA_RATIO
    image_center = (frame_width // 2, frame_height // 2)

    contours = _find_contours(binary_frame)

    candidates = []
    for contour in contours:
        candidate = _score_candidate(contour, max_allowed_area)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None, contours

    best_candidate = max(candidates, key=lambda item: (item["confidence"], item["area"]))
    if best_candidate["confidence"] < CONFIDENCE_THRESHOLD:
        return None, contours

    landing_point = best_candidate["landing_point"]
    if landing_point is None:
        return None, contours

    offset_x = landing_point[0] - image_center[0]
    offset_y = landing_point[1] - image_center[1]

    best_candidate["image_center"] = image_center
    best_candidate["offset_x"] = offset_x
    best_candidate["offset_y"] = offset_y

    return best_candidate, contours
