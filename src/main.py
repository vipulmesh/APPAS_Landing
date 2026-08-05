import time

import cv2

try:
    from config import CAMERA_INDEX, DISPLAY_WINDOW_NAME, FRAME_HEIGHT, FRAME_WIDTH, MASK_WINDOW_NAME
    from alignment import get_alignment_guidance, update_docking_state
    from detection import detect_landing_marker
    from preprocessing import preprocess_frame
    from utils import draw_detection
except ImportError:
    from .config import CAMERA_INDEX, DISPLAY_WINDOW_NAME, FRAME_HEIGHT, FRAME_WIDTH, MASK_WINDOW_NAME
    from .alignment import get_alignment_guidance, update_docking_state
    from .detection import detect_landing_marker
    from .preprocessing import preprocess_frame
    from .utils import draw_detection


def main():
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera index and permissions.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    previous_time = time.perf_counter()
    docking_state = "Searching"
    stable_alignment_frames = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            processed = preprocess_frame(frame)
            detection, _ = detect_landing_marker(frame, processed)

            if detection is None:
                alignment = {
                    "status": "Searching",
                    "instructions": [],
                    "aligned": False,
                }
                docking_state = "Searching"
                stable_alignment_frames = 0
            else:
                alignment = get_alignment_guidance(detection["offset_x"], detection["offset_y"])
                docking_state, stable_alignment_frames = update_docking_state(
                    detection,
                    alignment,
                    stable_alignment_frames,
                )

            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            previous_time = current_time

            annotated_frame = draw_detection(
                frame,
                detection,
                alignment=alignment,
                docking_state=docking_state,
                fps=fps,
            )

            processed_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if processed_display.shape[0] != annotated_frame.shape[0]:
                ratio = annotated_frame.shape[0] / processed_display.shape[0]
                new_width = int(processed_display.shape[1] * ratio)
                processed_display = cv2.resize(processed_display, (new_width, annotated_frame.shape[0]))

            display_frame = cv2.hconcat([annotated_frame, processed_display])

            cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
            cv2.imshow(MASK_WINDOW_NAME, processed)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
