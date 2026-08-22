from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from src.aruco.detector import ArucoDetector, DetectedMarker
from src.aruco.embedded import (
    detect_embedded_markers,
    find_valid_outer_ids,
    generate_embedded_marker,
    generate_standalone_inner_marker,
    inner_black_cell_ratio,
    interpret_embedded_markers,
    marker_quality,
    outer_center_is_black,
    validate_physical_dimensions,
    validate_embedded_configuration,
)


DICTIONARY = "DICT_7X7_100"
OUTER_ID = 25
INNER_ID = 45


def test_embedded_ids_must_differ():
    with pytest.raises(ValueError, match="must be different"):
        validate_embedded_configuration(DICTIONARY, OUTER_ID, OUTER_ID)


def test_outer_center_validation_and_valid_id_search():
    assert outer_center_is_black(DICTIONARY, OUTER_ID) is True
    assert outer_center_is_black(DICTIONARY, 23) is False
    assert OUTER_ID in find_valid_outer_ids(DICTIONARY)


def test_invalid_outer_id_is_rejected_before_image_generation():
    with TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="white central encoding cell"):
            generate_embedded_marker(DICTIONARY, 23, INNER_ID, 900, Path(directory) / "invalid.png", 0.5, 45)


def test_inner_black_ratio_and_validation():
    ratio = inner_black_cell_ratio(DICTIONARY, INNER_ID)
    assert ratio == pytest.approx(31 / 49)
    report = validate_embedded_configuration(DICTIONARY, OUTER_ID, INNER_ID, 0.5)
    assert report.valid is True
    assert report.inner_black_ratio == pytest.approx(ratio)
    assert validate_physical_dimensions(450, 90, 0.20) == pytest.approx(0.20)


def test_generated_embedded_marker_has_expected_geometry_and_detectable_ids():
    with TemporaryDirectory() as directory:
        path, report = generate_embedded_marker(DICTIONARY, OUTER_ID, INNER_ID, 1200, Path(directory) / "marker.png", 0.5, 60)
        image = cv2.imread(str(path))
        assert image.shape[:2] == (1200, 1200)
        assert report.valid is True
        direct_markers = ArucoDetector(DICTIONARY).detect(image)
        assert [marker.marker_id for marker in direct_markers] == [INNER_ID]
        markers = detect_embedded_markers(image, ArucoDetector(DICTIONARY), OUTER_ID, INNER_ID)
        by_id = {marker.marker_id: marker for marker in markers}
        assert set(by_id) >= {OUTER_ID, INNER_ID}
        assert by_id[INNER_ID].center == pytest.approx(by_id[OUTER_ID].center, abs=1)


def test_standalone_inner_marker_is_detected_by_normal_detector():
    with TemporaryDirectory() as directory:
        path = generate_standalone_inner_marker(DICTIONARY, INNER_ID, 1200, Path(directory) / "inner.png")
        assert [marker.marker_id for marker in ArucoDetector(DICTIONARY).detect(cv2.imread(str(path)))] == [INNER_ID]


@pytest.mark.parametrize("outer_id,inner_id", [(1, 45), (25, 45), (31, 45)])
def test_specialized_outer_decoder_works_for_multiple_valid_id_pairs(outer_id, inner_id):
    with TemporaryDirectory() as directory:
        path, _ = generate_embedded_marker(DICTIONARY, outer_id, inner_id, 1200, Path(directory) / "marker.png", 0.5, 60)
        image = cv2.imread(str(path))
        direct_ids = {marker.marker_id for marker in ArucoDetector(DICTIONARY).detect(image)}
        recovered_ids = {marker.marker_id for marker in detect_embedded_markers(image, ArucoDetector(DICTIONARY), outer_id, inner_id)}
        assert inner_id in direct_ids
        assert {outer_id, inner_id} <= recovered_ids


@pytest.mark.parametrize("destination", [
    [[100, 80], [1280, 110], [1240, 1260], [90, 1210]],
    [[140, 60], [1260, 180], [1120, 1280], [170, 1130]],
    [[230, 40], [1280, 250], [1040, 1320], [260, 1080]],
])
def test_direct_inner_and_specialized_outer_handle_moderate_perspective(destination):
    with TemporaryDirectory() as directory:
        path, _ = generate_embedded_marker(DICTIONARY, OUTER_ID, INNER_ID, 1200, Path(directory) / "marker.png", 0.5, 60)
        image = cv2.imread(str(path))
        transform = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1199, 0], [1199, 1199], [0, 1199]]),
            np.float32(destination),
        )
        warped = cv2.warpPerspective(image, transform, (1400, 1400), borderValue=(255, 255, 255))
        assert INNER_ID in {marker.marker_id for marker in ArucoDetector(DICTIONARY).detect(warped)}
        recovered = detect_embedded_markers(warped, ArucoDetector(DICTIONARY), OUTER_ID, INNER_ID)
        by_id = {marker.marker_id: marker for marker in recovered}
        assert {OUTER_ID, INNER_ID} <= set(by_id)
        assert marker_quality(by_id[INNER_ID]).geometry_valid is True


def test_embedded_marker_selection_policy():
    outer = DetectedMarker(OUTER_ID, [], (10, 10))
    inner = DetectedMarker(INNER_ID, [], (20, 20))
    assert interpret_embedded_markers([outer], OUTER_ID, INNER_ID).active_kind == "OUTER"
    assert interpret_embedded_markers([inner], OUTER_ID, INNER_ID).active_kind == "INNER"
    assert interpret_embedded_markers([outer, inner], OUTER_ID, INNER_ID).active_kind == "OUTER"
    assert interpret_embedded_markers([outer, inner], OUTER_ID, INNER_ID, "inner").active_kind == "INNER"
    assert interpret_embedded_markers([], OUTER_ID, INNER_ID).active is None
