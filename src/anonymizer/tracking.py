"""Temporal tracking to keep anonymization stable between detections.

Per-frame detection is unstable: a face the detector finds in one frame is
routinely missed in the next, so the region flickers and every dropout leaks
an unblurred frame. Linking detections across frames and holding a region for
a short while after its last sighting closes those gaps.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .detector import Detection


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two detection boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not intersection:
        return 0.0

    union = (
        (a.x2 - a.x1) * (a.y2 - a.y1)
        + (b.x2 - b.x1) * (b.y2 - b.y1)
        - intersection
    )
    return intersection / union if union else 0.0


@dataclass
class Track:
    """One object followed across frames."""

    detection: Detection
    last_seen: int
    hits: int = 1
    velocity: Tuple[float, float] = (0.0, 0.0)

    def predict(self, frame_number: int) -> Detection:
        """Where the object should be now, given how it was last moving.

        A held box that ignores motion anonymizes where the face *was*. Over
        a gap of several frames at walking pace that is far enough to expose
        the face it is meant to cover.
        """
        gap = frame_number - self.last_seen
        if not gap or self.velocity == (0.0, 0.0):
            return self.detection

        dx = int(self.velocity[0] * gap)
        dy = int(self.velocity[1] * gap)
        d = self.detection

        return Detection(
            x1=d.x1 + dx,
            y1=d.y1 + dy,
            x2=d.x2 + dx,
            y2=d.y2 + dy,
            confidence=d.confidence,
            class_name=d.class_name,
        )


class DetectionTracker:
    """Links detections across frames and bridges short dropouts."""

    def __init__(
        self,
        persistence: int = 10,
        iou_threshold: float = 0.3,
        max_velocity_frames: int = 5,
    ):
        """Create a tracker.

        Args:
            persistence: Frames to keep anonymizing a region after its last
                detection. Covers dropouts without holding stale boxes so long
                that they drift onto unrelated parts of the frame.
            iou_threshold: Overlap required to consider a detection a
                continuation of an existing track.
            max_velocity_frames: Gap beyond which a track's predicted motion is
                no longer trusted and the box stops being carried forward.
        """
        self.persistence = persistence
        self.iou_threshold = iou_threshold
        self.max_velocity_frames = max_velocity_frames
        self._tracks: List[Track] = []

    def reset(self) -> None:
        """Forget all tracks, for reuse across videos."""
        self._tracks = []

    def update(
        self, detections: List[Detection], frame_number: int
    ) -> List[Detection]:
        """Fold this frame's detections into the tracks.

        Args:
            detections: What the detector found in this frame.
            frame_number: Index of this frame.

        Returns:
            Regions to anonymize: everything detected now, plus recently seen
            regions the detector dropped.
        """
        unmatched = list(detections)
        surviving: List[Track] = []

        for track in self._tracks:
            predicted = track.predict(frame_number)

            best, best_score = None, self.iou_threshold
            for candidate in unmatched:
                score = iou(predicted, candidate)
                if score >= best_score:
                    best, best_score = candidate, score

            if best is not None:
                unmatched.remove(best)
                gap = max(1, frame_number - track.last_seen)
                track.velocity = (
                    (best.x1 - track.detection.x1) / gap,
                    (best.y1 - track.detection.y1) / gap,
                )
                track.detection = best
                track.last_seen = frame_number
                track.hits += 1
                surviving.append(track)
            elif frame_number - track.last_seen < self.persistence:
                surviving.append(track)

        for detection in unmatched:
            surviving.append(Track(detection=detection, last_seen=frame_number))

        self._tracks = surviving

        return [
            track.detection
            if track.last_seen == frame_number
            else track.predict(
                min(frame_number, track.last_seen + self.max_velocity_frames)
            )
            for track in self._tracks
        ]

    @property
    def active_tracks(self) -> int:
        """How many regions are currently being followed."""
        return len(self._tracks)
