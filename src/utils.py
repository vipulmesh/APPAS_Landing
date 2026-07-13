import cv2

try:
    from config import BOX_COLOR, CONTOUR_COLOR, FONT, FONT_SCALE, FONT_THICKNESS, STATUS_COLOR, TEXT_COLOR
except ImportError:
    from .config import BOX_COLOR, CONTOUR_COLOR, FONT, FONT_SCALE, FONT_THICKNESS, STATUS_COLOR, TEXT_COLOR


def draw_detection(frame, detection):
    output = frame.copy()

    if detection is None:
        cv2.putText(output, "Status: Searching", (20, 40), FONT, FONT_SCALE, STATUS_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        return output

    contour, (x, y, w, h), center, area = detection

    cv2.drawContours(output, [contour], -1, CONTOUR_COLOR, 2)
    cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR, 2)

    if center is not None:
        cv2.circle(output, center, 5, (0, 0, 255), -1)
        cv2.putText(output, f"Center: {center[0]}, {center[1]}", (x, max(25, y - 10)), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    cv2.putText(output, f"Area: {int(area)}", (20, 40), FONT, FONT_SCALE, STATUS_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return output
