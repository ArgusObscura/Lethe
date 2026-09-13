"""Plate detection must not anonymize arbitrary objects by default."""

import pytest

from anonymizer.models.config import AnonymizationConfig
from anonymizer.models.loader import ModelLoader


class FakeModel:
    def __init__(self, names):
        self.names = names


COCO_SAMPLE = {0: "person", 1: "bicycle", 2: "car", 9: "traffic light"}


class TestPlateModelRecognition:
    def test_general_purpose_model_is_not_a_plate_detector(self):
        assert ModelLoader._detects_license_plates(FakeModel(COCO_SAMPLE)) is False

    def test_plate_model_is_recognized(self):
        assert ModelLoader._detects_license_plates(FakeModel({0: "License_Plate"}))

    def test_recognition_is_case_insensitive(self):
        assert ModelLoader._detects_license_plates(FakeModel({0: "license plate"}))
        assert ModelLoader._detects_license_plates(FakeModel({0: "PLATE"}))

    def test_model_without_names_is_not_a_plate_detector(self):
        assert ModelLoader._detects_license_plates(object()) is False


class TestDefaults:
    def test_plate_detection_is_off_by_default(self):
        """On by default, the bundled model anonymizes every object it knows."""
        assert AnonymizationConfig().enable_license_plate_detection is False

    def test_face_detection_remains_on_by_default(self):
        assert AnonymizationConfig().enable_face_detection is True

    def test_plate_detection_can_still_be_enabled(self):
        config = AnonymizationConfig(enable_license_plate_detection=True)

        assert config.enable_license_plate_detection is True
