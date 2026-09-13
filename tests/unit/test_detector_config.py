"""Detection settings must reach the detector they were written for."""

import numpy as np
import pytest

from anonymizer.core import AnonymizationPipeline
from anonymizer.detector import ObjectDetector
from anonymizer.models.config import AnonymizationConfig, DetectionConfig


class RecordingModel:
    """Captures the keyword arguments each inference call was made with."""

    def __init__(self):
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return []


@pytest.fixture
def detector():
    det = ObjectDetector(
        model_loader=None,
        config=DetectionConfig(confidence_threshold=0.9, device="cpu"),
        license_plate_config=DetectionConfig(confidence_threshold=0.1, device="cpu"),
    )
    det.face_model = RecordingModel()
    det.lp_model = RecordingModel()
    return det


class TestSeparateThresholds:
    def test_faces_use_the_face_threshold(self, detector):
        detector.detect_faces(np.zeros((10, 10, 3), np.uint8))

        assert detector.face_model.calls[0]["conf"] == 0.9

    def test_plates_use_the_plate_threshold(self, detector):
        """Previously the detector held one config and used it for both."""
        detector.detect_license_plates(np.zeros((10, 10, 3), np.uint8))

        assert detector.lp_model.calls[0]["conf"] == 0.1

    def test_batched_calls_use_the_same_thresholds(self, detector):
        frames = [np.zeros((10, 10, 3), np.uint8)] * 3
        detector.detect_faces_batch(frames)
        detector.detect_license_plates_batch(frames)

        assert detector.face_model.calls[0]["conf"] == 0.9
        assert detector.lp_model.calls[0]["conf"] == 0.1

    def test_plate_config_defaults_to_face_config(self):
        det = ObjectDetector(
            model_loader=None,
            config=DetectionConfig(confidence_threshold=0.7),
        )

        assert det.license_plate_config.confidence_threshold == 0.7


class TestPipelineWiring:
    def test_pipeline_passes_both_configs_through(self):
        pipeline = AnonymizationPipeline(
            AnonymizationConfig(
                face_config=DetectionConfig(confidence_threshold=0.9),
                license_plate_config=DetectionConfig(confidence_threshold=0.1),
            )
        )

        assert pipeline.detector.config.confidence_threshold == 0.9
        assert pipeline.detector.license_plate_config.confidence_threshold == 0.1
