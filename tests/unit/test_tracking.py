"""Tests for temporal tracking of detections."""

import pytest

from anonymizer.detector import Detection
from anonymizer.tracking import DetectionTracker, Track, iou


def box(x1, y1, x2, y2, conf=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_name="face")


class TestIou:
    def test_identical_boxes(self):
        assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou(box(0, 0, 10, 10), box(50, 50, 60, 60)) == 0.0

    def test_touching_boxes_do_not_overlap(self):
        assert iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0

    def test_partial_overlap(self):
        assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1 / 3)


class TestBridgingDropouts:
    """The defect this exists for: a detector dropout leaks an unblurred frame."""

    def test_region_is_held_through_a_gap(self):
        tracker = DetectionTracker(persistence=10)

        tracker.update([box(100, 100, 140, 150)], 0)
        held = tracker.update([], 1)

        assert len(held) == 1, "face vanished the moment detection missed it"

    def test_region_is_held_across_many_missed_frames(self):
        tracker = DetectionTracker(persistence=10)
        tracker.update([box(100, 100, 140, 150)], 0)

        for frame in range(1, 10):
            assert len(tracker.update([], frame)) == 1, f"lost at frame {frame}"

    def test_region_is_eventually_released(self):
        tracker = DetectionTracker(persistence=5)
        tracker.update([box(100, 100, 140, 150)], 0)

        for frame in range(1, 6):
            tracker.update([], frame)

        assert tracker.update([], 6) == []

    def test_redetection_refreshes_the_track(self):
        tracker = DetectionTracker(persistence=3)
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([], 1)
        tracker.update([box(100, 100, 140, 150)], 2)

        # The re-detection resets the clock, so it survives past 0 + 3.
        assert len(tracker.update([], 4)) == 1

    def test_persistence_zero_disables_bridging(self):
        tracker = DetectionTracker(persistence=0)
        tracker.update([box(100, 100, 140, 150)], 0)

        assert tracker.update([], 1) == []


class TestIdentity:
    def test_overlapping_detection_continues_one_track(self):
        tracker = DetectionTracker()
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([box(104, 102, 144, 152)], 1)

        assert tracker.active_tracks == 1

    def test_distant_detection_starts_a_new_track(self):
        tracker = DetectionTracker()
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([box(600, 600, 640, 650)], 1)

        assert tracker.active_tracks == 2

    def test_two_faces_stay_separate(self):
        tracker = DetectionTracker()
        faces = [box(100, 100, 140, 150), box(400, 100, 440, 150)]

        tracker.update(faces, 0)
        regions = tracker.update(faces, 1)

        assert tracker.active_tracks == 2
        assert len(regions) == 2

    def test_one_detection_cannot_claim_two_tracks(self):
        tracker = DetectionTracker()
        tracker.update([box(100, 100, 140, 150), box(120, 100, 160, 150)], 0)

        tracker.update([box(100, 100, 140, 150)], 1)

        # The unmatched track is held rather than deleted or duplicated.
        assert tracker.active_tracks == 2


class TestMotion:
    def test_held_box_follows_the_motion_it_had(self):
        """A stale box anonymizes where the face was, not where it is."""
        tracker = DetectionTracker(persistence=10)
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([box(110, 100, 150, 150)], 1)  # moving +10px/frame

        held = tracker.update([], 2)[0]

        assert held.x1 > 110, f"box did not advance with the subject: {held.x1}"

    def test_stationary_track_does_not_drift(self):
        tracker = DetectionTracker(persistence=10)
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([box(100, 100, 140, 150)], 1)

        held = tracker.update([], 2)[0]

        assert held.x1 == 100

    def test_extrapolation_is_bounded(self):
        """Motion is not trusted indefinitely, or the box flies off-frame."""
        tracker = DetectionTracker(persistence=30, max_velocity_frames=5)
        tracker.update([box(100, 100, 140, 150)], 0)
        tracker.update([box(150, 100, 190, 150)], 1)  # 50px/frame

        far = tracker.update([], 25)[0]

        assert far.x1 <= 150 + 50 * 5


class TestReset:
    def test_reset_clears_tracks(self):
        tracker = DetectionTracker()
        tracker.update([box(100, 100, 140, 150)], 0)

        tracker.reset()

        assert tracker.active_tracks == 0
        assert tracker.update([], 1) == []


class TestPassThrough:
    def test_detections_are_returned_on_the_frame_they_arrive(self):
        tracker = DetectionTracker()
        faces = [box(100, 100, 140, 150)]

        assert tracker.update(faces, 0) == faces

    def test_no_detections_and_no_history_yields_nothing(self):
        assert DetectionTracker().update([], 0) == []
