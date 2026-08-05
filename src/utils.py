import cv2

try:
    from config import (
        ALIGNED_COLOR,
        BOX_COLOR,
        BOX_THICKNESS,
        CONTOUR_COLOR,
        CONTOUR_THICKNESS,
        CENTER_LABEL_OFFSET_X,
        CENTER_LABEL_OFFSET_Y,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
        CROSSHAIR_ARM_LENGTH,
        IMAGE_CENTER_COLOR,
        IMAGE_CENTER_RADIUS,
        IMAGE_CENTER_LINE_THICKNESS,
        LANDING_POINT_COLOR,
        LINE_COLOR,
        LINE_THICKNESS,
        LABEL_FONT_SCALE,
        LABEL_TEXT_MIN_Y,
        LABEL_THICKNESS,
        POINT_RADIUS,
        SEARCH_COLOR,
        STATUS_COLOR,
        STATUS_LINE_HEIGHT,
        STATUS_PANEL_X,
        STATUS_PANEL_Y,
        TEXT_COLOR,
        WARNING_COLOR,
    )
except ImportError:
    from .config import (
        ALIGNED_COLOR,
        BOX_COLOR,
        BOX_THICKNESS,
        CONTOUR_COLOR,
        CONTOUR_THICKNESS,
        CENTER_LABEL_OFFSET_X,
        CENTER_LABEL_OFFSET_Y,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
        CROSSHAIR_ARM_LENGTH,
        IMAGE_CENTER_COLOR,
        IMAGE_CENTER_RADIUS,
        IMAGE_CENTER_LINE_THICKNESS,
        LANDING_POINT_COLOR,
        LINE_COLOR,
        LINE_THICKNESS,
        LABEL_FONT_SCALE,
        LABEL_TEXT_MIN_Y,
        LABEL_THICKNESS,
        POINT_RADIUS,
        SEARCH_COLOR,
        STATUS_COLOR,
        STATUS_LINE_HEIGHT,
        STATUS_PANEL_X,
        STATUS_PANEL_Y,
        TEXT_COLOR,
        WARNING_COLOR,
    )


def _draw_text_lines(output, lines, start_x, start_y, color=STATUS_COLOR):
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (start_x, start_y + (index * STATUS_LINE_HEIGHT)),
            FONT,
            FONT_SCALE,
            color,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


def _draw_crosshair(output, point, color):
    x, y = point
    cv2.circle(output, point, IMAGE_CENTER_RADIUS, color, IMAGE_CENTER_LINE_THICKNESS)
    cv2.line(output, (x - CROSSHAIR_ARM_LENGTH, y), (x + CROSSHAIR_ARM_LENGTH, y), color, IMAGE_CENTER_LINE_THICKNESS)
    cv2.line(output, (x, y - CROSSHAIR_ARM_LENGTH), (x, y + CROSSHAIR_ARM_LENGTH), color, IMAGE_CENTER_LINE_THICKNESS)


def draw_detection(frame, detection, alignment=None, docking_state="Searching", fps=None):
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    image_center = (frame_width // 2, frame_height // 2)

    _draw_crosshair(output, image_center, IMAGE_CENTER_COLOR)
    cv2.putText(
        output,
        "Image Center",
        (image_center[0] + CENTER_LABEL_OFFSET_X, image_center[1] - CENTER_LABEL_OFFSET_Y),
        FONT,
        LABEL_FONT_SCALE,
        IMAGE_CENTER_COLOR,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )

    status_lines = [f"Docking State: {docking_state}"]

    if fps is not None:
        status_lines.append(f"FPS: {fps:.2f}")

    if detection is None:
        status_lines.append("Status: Searching")
        _draw_text_lines(output, status_lines, STATUS_PANEL_X, STATUS_PANEL_Y, SEARCH_COLOR)
        return output

    contour = detection["contour"]
    x, y, w, h = detection["bbox"]
    center = detection["center"]
    landing_point = detection["landing_point"]
    confidence = detection["confidence"]
    offset_x = detection["offset_x"]
    offset_y = detection["offset_y"]
    area = detection["area"]
    vertices = detection["vertices"]
    aspect_ratio = detection["aspect_ratio"]
    solidity = detection["solidity"]
    extent = detection["extent"]
    is_convex = detection["is_convex"]

    cv2.drawContours(output, [contour], -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
    cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

    if center is not None:
        cv2.circle(output, center, POINT_RADIUS, TEXT_COLOR, -1)
        cv2.putText(
            output,
            f"Marker Center: {center[0]}, {center[1]}",
            (x, max(LABEL_TEXT_MIN_Y, y - CENTER_LABEL_OFFSET_Y)),
            FONT,
            LABEL_FONT_SCALE,
            TEXT_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    if landing_point is not None:
        cv2.circle(output, landing_point, POINT_RADIUS, LANDING_POINT_COLOR, -1)
        cv2.line(output, image_center, landing_point, LINE_COLOR, LINE_THICKNESS)

    if alignment is None:
        alignment_status = "Unknown"
        instructions_text = "N/A"
    else:
        alignment_status = alignment["status"]
        instructions = alignment["instructions"]
        instructions_text = " | ".join(instructions) if instructions else "Hold Position"

    if docking_state == "Docked":
        state_color = ALIGNED_COLOR
    elif docking_state == "Searching":
        state_color = SEARCH_COLOR
    else:
        state_color = WARNING_COLOR

    status_lines.extend(
        [
            f"Confidence: {confidence:.2f}",
            f"Landing Point: {landing_point[0]}, {landing_point[1]}" if landing_point is not None else "Landing Point: N/A",
            f"Offset X: {offset_x}",
            f"Offset Y: {offset_y}",
            f"Alignment: {alignment_status}",
            f"Instruction: {instructions_text}",
            f"Area: {int(area)}",
            f"Vertices: {vertices}",
            f"Aspect Ratio: {aspect_ratio:.2f}",
            f"Solidity: {solidity:.2f}",
            f"Extent: {extent:.2f}",
            f"Convex: {'Yes' if is_convex else 'No'}",
        ]
    )

    _draw_text_lines(output, status_lines[:2], STATUS_PANEL_X, STATUS_PANEL_Y, state_color)
    _draw_text_lines(output, status_lines[2:], STATUS_PANEL_X, STATUS_PANEL_Y + (STATUS_LINE_HEIGHT * 2), TEXT_COLOR)

    return output
