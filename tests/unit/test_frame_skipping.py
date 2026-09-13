"""Detection can skip frames, but only where tracking covers the gap."""

import numpy as np
import pytest

from anonymizer.core import AnonymizationPipeline
from anonymizer.detector import Detection
from anonymizer.models.config import AnonymizationConfig


def frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


class RecordingDetector:
    """Counts how many frames were actually handed to the model."""

    def __init__(self):
        self.faces_seen = 0
        self.plates_seen = 0

    def detect_faces_batch(self, frames):
        self.faces_seen += len(frames)
        return [[Detection(x1=1, y1=1, x2=9, y2=9, confidence=0.9, class_name="face")]
                for _ in frames]

    def detect_license_plates_batch(self, frames):
        self.plates_seen += len(frames)
        return [[] for _ in frames]


def pipeline_with(step, **kwargs):
    config = AnonymizationConfig(detect_every=step, **kwargs)
    pipeline = AnonymizationPipeline(config)
    pipeline.detector = RecordingDetector()
    return pipeline


class TestValidation:
    def test_skipping_without_tracking_is_rejected(self):
        """Otherwise only every Nth frame would be anonymized at all."""
        with pytest.raises(ValueError, match="track_detections"):
            AnonymizationConfig(detect_every=3, track_detections=False)

    def test_no_skipping_without_tracking_is_fine(self):
        config = AnonymizationConfig(detect_every=1, track_detections=False)
        assert config.detect_every == 1

    def test_detect_every_must_be_positive(self):
        with pytest.raises(ValueError):
            AnonymizationConfig(detect_every=0)


class TestSelection:
    @pytest.mark.parametrize("step,expected", [(1, 8), (2, 4), (4, 2), (8, 1)])
    def test_only_every_nth_frame_reaches_the_detector(self, step, expected):
        pipeline = pipeline_with(step)
        batch = [(i, frame()) for i in range(8)]

        pipeline._detect_batch(batch)

        assert pipeline.detector.faces_seen == expected

    def test_results_land_on_the_frames_they_came_from(self):
        pipeline = pipeline_with(2)
        batch = [(i, frame()) for i in range(6)]

        faces, _ = pipeline._detect_batch(batch)

        assert [bool(f) for f in faces] == [True, False, True, False, True, False]

    def test_output_has_one_entry_per_input_frame(self):
        pipeline = pipeline_with(3)
        batch = [(i, frame()) for i in range(7)]

        faces, plates = pipeline._detect_batch(batch)

        assert len(faces) == 7 and len(plates) == 7

    def test_skipping_follows_absolute_frame_numbers(self):
        """Batches do not start at zero, so selection cannot be batch-relative."""
        pipeline = pipeline_with(4)
        batch = [(i, frame()) for i in range(100, 108)]

        faces, _ = pipeline._detect_batch(batch)

        # 100 and 104 are the multiples of 4 in this range.
        assert [bool(f) for f in faces] == [
            True, False, False, False, True, False, False, False
        ]

    def test_disabled_detectors_are_never_called(self):
        pipeline = pipeline_with(1, enable_face_detection=False,
                                 enable_license_plate_detection=False)
        batch = [(i, frame()) for i in range(4)]

        faces, plates = pipeline._detect_batch(batch)

        assert pipeline.detector.faces_seen == 0
        assert faces == [[], [], [], []] and plates == [[], [], [], []]


class TestPersistence:
    def test_track_persistence_outlives_the_detection_gap(self):
        """A track dying between detections would strobe the anonymization."""
        pipeline = AnonymizationPipeline(
            AnonymizationConfig(detect_every=8, track_persistence=10)
        )

        assert pipeline.face_tracker.persistence >= 8

    def test_explicit_persistence_is_respected_when_larger(self):
        pipeline = AnonymizationPipeline(
            AnonymizationConfig(detect_every=2, track_persistence=40)
        )

        assert pipeline.face_tracker.persistence == 40
