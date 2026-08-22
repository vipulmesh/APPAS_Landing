from src.aruco.detector import ArucoDetector, DetectedMarker
from src.aruco.localization import alignment_error, classify_alignment, frame_center, marker_center


def test_marker_center_is_average_of_four_corners():
    assert marker_center([(100, 100), (200, 100), (200, 200), (100, 200)]) == (150, 150)


def test_frame_center_adapts_to_resolution():
    assert frame_center(640, 480) == (320, 240)


def test_alignment_error_is_marker_minus_frame_center():
    assert alignment_error((400, 300), (320, 240)) == (80, 60)


def test_centered_marker_is_aligned():
    result = classify_alignment(0, 0, 20, 20)
    assert result.aligned is True
    assert result.status == "ALIGNED"


def test_directional_alignment_cases():
    assert classify_alignment(-21, 0, 20, 20).status == "LEFT"
    assert classify_alignment(21, 0, 20, 20).status == "RIGHT"
    assert classify_alignment(0, -21, 20, 20).status == "UP"
    assert classify_alignment(0, 21, 20, 20).status == "DOWN"
    assert classify_alignment(21, 21, 20, 20).status == "RIGHT / DOWN"


def test_tolerance_boundaries_are_aligned_until_exceeded():
    assert classify_alignment(20, -20, 20, 20).aligned is True
    assert classify_alignment(19, -19, 20, 20).aligned is True
    assert classify_alignment(20.1, 0, 20, 20).status == "RIGHT"
    assert classify_alignment(0, -20.1, 20, 20).status == "UP"


def test_target_marker_selection_does_not_use_first_marker():
    markers = [DetectedMarker(marker_id, [], (0, 0)) for marker_id in (5, 12, 23, 40)]
    assert ArucoDetector.select_target(markers, 23).marker_id == 23
    assert ArucoDetector.select_target(markers, 99) is None
