"""Out-of-the-box settings must actually find things."""

import numpy as np
import pytest

from anonymizer.detector import ObjectDetector
from anonymizer.models.config import AnonymizationConfig, DetectionConfig


class TestAutoInferenceSize:
    """A fixed 640 silently downscales anything larger, losing small faces."""

    @pytest.mark.parametrize("width,expected", [
        (640, 640),
        (1280, 1280),
        (1920, 1920),
    ])
    def test_follows_the_footage_width(self, width, expected):
        assert ObjectDetector.auto_inference_size(width) == expected

    def test_capped_for_very_large_footage(self):
        assert ObjectDetector.auto_inference_size(2730) == 1920
        assert ObjectDetector.auto_inference_size(7680) == 1920

    def test_floored_for_small_footage(self):
        assert ObjectDetector.auto_inference_size(320) == 640

    def test_always_a_multiple_of_32(self):
        """YOLO's stride; a non-multiple gets silently rounded anyway."""
        for width in range(640, 1960, 17):
            assert ObjectDetector.auto_inference_size(width) % 32 == 0


class TestSizeSelection:
    def frames(self, width, height=720):
        return [np.zeros((height, width, 3), dtype=np.uint8)]

    def test_explicit_size_wins(self):
        config = DetectionConfig(inference_size=800)

        assert ObjectDetector._size_kwargs(config, self.frames(2730)) == {"imgsz": 800}

    def test_unset_size_follows_the_frame(self):
        config = DetectionConfig()

        assert ObjectDetector._size_kwargs(config, self.frames(1280)) == {"imgsz": 1280}

    def test_large_frame_is_capped(self):
        config = DetectionConfig()

        assert ObjectDetector._size_kwargs(config, self.frames(2730)) == {"imgsz": 1920}

    def test_without_frames_nothing_is_forced(self):
        assert ObjectDetector._size_kwargs(DetectionConfig(), None) == {}
        assert ObjectDetector._size_kwargs(DetectionConfig(), []) == {}


class TestDefaults:
    def test_inference_size_is_unset_so_it_auto_scales(self):
        assert DetectionConfig().inference_size is None

    def test_confidence_favours_recall(self):
        """Missing a face is a privacy failure; an extra blur is cosmetic."""
        assert DetectionConfig().confidence_threshold == 0.35

    def test_protections_are_on_by_default(self):
        config = AnonymizationConfig()

        assert config.enable_face_detection is True
        assert config.enable_license_plate_detection is True
        assert config.person_gate is True
        assert config.track_detections is True

    def test_no_frames_are_skipped_by_default(self):
        assert AnonymizationConfig().detect_every == 1
