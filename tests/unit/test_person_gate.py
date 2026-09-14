"""Large face detections must be backed by a person."""

import pytest

from anonymizer.detector import Detection, ObjectDetector
from anonymizer.models.config import AnonymizationConfig


def face(x1, y1, x2, y2):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.6, class_name="face")


def person(x1, y1, x2, y2):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.8, class_name="person")


gate = ObjectDetector.gate_faces_by_person


class TestLargeDetections:
    """The defect: car wheels and rear ends read as confident, large faces."""

    def test_large_face_without_a_person_is_dropped(self):
        wheel = face(100, 100, 200, 280)  # 100px wide, no person anywhere

        assert gate([wheel], [], min_size=60) == []

    def test_large_face_on_a_person_is_kept(self):
        head = face(100, 100, 180, 200)
        walker = person(80, 90, 220, 600)

        assert gate([head], [walker], min_size=60) == [head]

    def test_person_elsewhere_does_not_rescue_a_wheel(self):
        wheel = face(100, 100, 200, 280)
        walker = person(900, 100, 1000, 600)

        assert gate([wheel], [walker], min_size=60) == []

    def test_partial_overlap_is_not_enough(self):
        """A face clipping the edge of a person is not on that person."""
        wheel = face(100, 100, 200, 200)
        walker = person(180, 180, 400, 600)

        assert gate([wheel], [walker], min_size=60) == []


class TestSmallDetections:
    """Person detection is unreliable at distance, so small faces pass."""

    def test_small_face_is_kept_without_a_person(self):
        distant = face(100, 100, 130, 140)  # 30px wide

        assert gate([distant], [], min_size=60) == [distant]

    def test_threshold_is_on_width(self):
        just_under = face(0, 0, 59, 200)
        just_over = face(0, 0, 61, 200)

        assert gate([just_under], [], min_size=60) == [just_under]
        assert gate([just_over], [], min_size=60) == []


class TestMixedFrames:
    def test_real_faces_survive_while_wheels_are_dropped(self):
        walker = person(500, 200, 620, 700)
        real = face(530, 220, 600, 300)
        wheel = face(100, 400, 220, 560)
        distant = face(800, 300, 820, 325)

        kept = gate([real, wheel, distant], [walker], min_size=60)

        assert real in kept and distant in kept
        assert wheel not in kept

    def test_no_faces_yields_nothing(self):
        assert gate([], [person(0, 0, 100, 300)], min_size=60) == []

    def test_degenerate_box_passes_through_harmlessly(self):
        """Zero-width boxes are below any threshold and anonymize nothing."""
        empty = face(50, 50, 50, 50)

        assert gate([empty], [], min_size=60) == [empty]

    def test_large_degenerate_box_needs_support_like_any_other(self):
        flat = face(50, 50, 200, 50)  # wide but zero height

        assert gate([flat], [person(0, 0, 500, 500)], min_size=60) == []


class TestConfig:
    def test_gating_is_on_by_default(self):
        assert AnonymizationConfig().person_gate is True

    def test_default_threshold(self):
        assert AnonymizationConfig().person_gate_min_size == 60

    def test_gating_can_be_disabled(self):
        assert AnonymizationConfig(person_gate=False).person_gate is False
