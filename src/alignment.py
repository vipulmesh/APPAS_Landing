try:
    from config import (
        ALIGNMENT_TOLERANCE,
        CENTER_TOLERANCE,
        DOCKED_CONFIRMATION_FRAMES,
        READY_TO_LAND_CONFIDENCE,
    )
except ImportError:
    from .config import (
        ALIGNMENT_TOLERANCE,
        CENTER_TOLERANCE,
        DOCKED_CONFIRMATION_FRAMES,
        READY_TO_LAND_CONFIDENCE,
    )


def get_alignment_guidance(offset_x, offset_y):
    instructions = []

    if offset_x > ALIGNMENT_TOLERANCE:
        instructions.append("Move Right")
    elif offset_x < -ALIGNMENT_TOLERANCE:
        instructions.append("Move Left")

    if offset_y > ALIGNMENT_TOLERANCE:
        instructions.append("Move Down")
    elif offset_y < -ALIGNMENT_TOLERANCE:
        instructions.append("Move Up")

    aligned = not instructions
    status = "Aligned" if aligned else "Adjust Position"

    return {
        "status": status,
        "instructions": instructions,
        "aligned": aligned,
    }


def update_docking_state(detection, alignment, stable_alignment_frames):
    if detection is None:
        return "Searching", 0

    confidence = detection["confidence"]
    offset_x = detection["offset_x"]
    offset_y = detection["offset_y"]

    if not alignment["aligned"]:
        return "Aligning", 0

    if abs(offset_x) > CENTER_TOLERANCE or abs(offset_y) > CENTER_TOLERANCE:
        return "Marker Detected", 0

    if confidence >= READY_TO_LAND_CONFIDENCE:
        stable_alignment_frames += 1
        if stable_alignment_frames >= DOCKED_CONFIRMATION_FRAMES:
            return "Docked", stable_alignment_frames
        return "Ready to Land", stable_alignment_frames

    return "Marker Detected", 0
