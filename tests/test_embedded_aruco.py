from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import pytest

from src.aruco.detector import ArucoDetector, DetectedMarker
from src.aruco.embedded import (
    detect_embedded_markers,
    find_valid_outer_ids,
    generate_embedded_marker,
    inner_black_cell_ratio,
    interpret_embedded_markers,
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
    assert validate_physical_dimensions(450, 50) == pytest.approx(1 / 9)


def test_generated_embedded_marker_has_expected_geometry_and_detectable_ids():
    with TemporaryDirectory() as directory:
        path, report = generate_embedded_marker(DICTIONARY, OUTER_ID, INNER_ID, 900, Path(directory) / "marker.png", 0.5, 45)
        image = cv2.imread(str(path))
        assert image.shape[:2] == (900, 900)
        assert report.valid is True
        markers = detect_embedded_markers(image, ArucoDetector(DICTIONARY), OUTER_ID, INNER_ID)
        by_id = {marker.marker_id: marker for marker in markers}
        assert set(by_id) >= {OUTER_ID, INNER_ID}
        assert by_id[INNER_ID].center == pytest.approx(by_id[OUTER_ID].center, abs=1)


def test_embedded_marker_selection_policy():
    outer = DetectedMarker(OUTER_ID, [], (10, 10))
    inner = DetectedMarker(INNER_ID, [], (20, 20))
    assert interpret_embedded_markers([outer], OUTER_ID, INNER_ID).active_kind == "OUTER"
    assert interpret_embedded_markers([inner], OUTER_ID, INNER_ID).active_kind == "INNER"
    assert interpret_embedded_markers([outer, inner], OUTER_ID, INNER_ID).active_kind == "OUTER"
    assert interpret_embedded_markers([outer, inner], OUTER_ID, INNER_ID, "inner").active_kind == "INNER"
    assert interpret_embedded_markers([], OUTER_ID, INNER_ID).active is None
