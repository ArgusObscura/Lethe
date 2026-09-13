"""Regression tests for anonymization correctness and privacy strength."""

import cv2
import numpy as np
import pytest

from anonymizer.anonymizer import Anonymizer
from anonymizer.detector import Detection
from anonymizer.models.config import AnonymizationConfig


def detection(x1, y1, x2, y2):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_name="face")


@pytest.fixture
def frame():
    """A frame carrying low-frequency structure.

    Random noise is useless for testing anonymization strength: it is pure
    high frequency, so any blur destroys it and every kernel looks perfect.
    Identity lives in coarse structure, so this upsamples a small random
    field into smooth large-scale features instead.
    """
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 255, (10, 14, 3), dtype=np.uint8)
    return cv2.resize(coarse, (800, 600), interpolation=cv2.INTER_CUBIC)


def structure_correlation(original, processed):
    """How much coarse structure survives: 1.0 = intact, 0.0 = destroyed."""
    def signature(img):
        small = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16)
        ).astype(float)
        return (small - small.mean()) / (small.std() + 1e-6)

    return float((signature(original) * signature(processed)).mean())


class TestPixelateSmallRegions:
    """A region narrower than one block used to crash the intermediate resize."""

    @pytest.mark.parametrize("width,height", [(1, 1), (3, 5), (13, 29), (14, 14)])
    def test_regions_smaller_than_block_size_do_not_crash(self, frame, width, height):
        anonymizer = Anonymizer(AnonymizationConfig(method="pixelate", pixelate_size=15))

        result = anonymizer.anonymize_frame(frame, [detection(10, 10, 10 + width, 10 + height)])

        assert result.shape == frame.shape

    def test_tiny_region_is_actually_flattened(self, frame):
        anonymizer = Anonymizer(
            AnonymizationConfig(method="pixelate", pixelate_size=15, box_padding=0.0)
        )

        result = anonymizer.anonymize_frame(frame, [detection(10, 10, 22, 22)])

        # Collapsed to a single block, so every pixel in the region matches.
        region = result[10:22, 10:22]
        assert len(np.unique(region.reshape(-1, 3), axis=0)) == 1


@pytest.fixture
def face_pattern():
    """Stand-in for a face: structure at several spatial scales."""
    rng = np.random.default_rng(7)
    coarse = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(coarse, (512, 512), interpolation=cv2.INTER_CUBIC)


def render_at(pattern, size):
    """Put the same pattern on screen at a given size, with a margin."""
    roi = cv2.resize(pattern, (size, size), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size + 40, size + 40, 3), dtype=np.uint8)
    canvas[20:20 + size, 20:20 + size] = roi
    return canvas, roi


def survival_by_size(method, sizes, pattern):
    """Structure surviving anonymization when the same face is at each size.

    Scaling one pattern keeps the content identical in relative terms, so any
    difference between sizes comes from the anonymization, not the imagery.
    """
    anonymizer = Anonymizer(AnonymizationConfig(method=method, box_padding=0.0))
    out = {}

    for size in sizes:
        canvas, roi = render_at(pattern, size)
        result = anonymizer.anonymize_frame(
            canvas, [detection(20, 20, 20 + size, 20 + size)]
        )
        out[size] = structure_correlation(roi, result[20:20 + size, 20:20 + size])

    return out


class TestStrengthIsSizeIndependent:
    """The core privacy defect: a fixed kernel or block size anonymizes a
    close-up face far more weakly than a distant one, which is backwards."""

    SIZES = (40, 120, 300, 500)

    def test_fixed_kernel_would_fail_this(self, face_pattern):
        """Guards the premise: the old approach really does degrade."""
        survival = {}
        for size in self.SIZES:
            _, roi = render_at(face_pattern, size)
            survival[size] = structure_correlation(
                roi, cv2.GaussianBlur(roi, (31, 31), 0)
            )

        assert survival[500] > survival[40] + 0.1, (
            "expected the fixed kernel to weaken on large regions, got "
            f"{survival}"
        )

    def test_blur_strength_does_not_weaken_as_regions_grow(self, face_pattern):
        survival = survival_by_size("blur", self.SIZES, face_pattern)

        assert survival[500] <= survival[40] + 0.05, survival

    def test_pixelate_strength_does_not_weaken_as_regions_grow(self, face_pattern):
        """Compared across sizes large enough to resolve the pattern.

        A 40px region has fewer pixels than blocks, so it is inherently
        flattened further; that is not the defect being guarded here.
        """
        survival = survival_by_size("pixelate", (120, 300, 500), face_pattern)

        assert survival[500] <= survival[120] + 0.05, survival

    def test_uncapped_blocks_would_fail_this(self, face_pattern):
        """Guards the premise: fixed block size really does weaken with size."""
        survival = {}
        for size in (120, 500):
            _, roi = render_at(face_pattern, size)
            blocks = max(1, size // 15)  # the old, uncapped behaviour
            small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            restored = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
            survival[size] = structure_correlation(roi, restored)

        assert survival[500] > survival[120] + 0.05, survival

    def test_pixelate_block_count_is_capped(self, frame):
        """A large region must not subdivide into fine, identifiable blocks."""
        anonymizer = Anonymizer(
            AnonymizationConfig(method="pixelate", pixelate_size=15, box_padding=0.0)
        )

        result = anonymizer.anonymize_frame(frame, [detection(20, 20, 620, 520)])

        region = result[20:520, 20:620]
        distinct = len(np.unique(region.reshape(-1, 3), axis=0))
        assert distinct <= Anonymizer.MAX_PIXELATE_BLOCKS ** 2, distinct


class TestBoxPadding:
    def test_padding_covers_beyond_the_detector_box(self, frame):
        anonymizer = Anonymizer(AnonymizationConfig(method="mask", box_padding=0.25))

        result = anonymizer.anonymize_frame(frame, [detection(200, 200, 300, 300)])

        # 25% of a 100px box reaches 25px outside it.
        assert np.all(result[190:310, 190:310] == 0)

    def test_zero_padding_leaves_the_box_exact(self, frame):
        anonymizer = Anonymizer(AnonymizationConfig(method="mask", box_padding=0.0))

        result = anonymizer.anonymize_frame(frame, [detection(200, 200, 300, 300)])

        assert np.all(result[200:300, 200:300] == 0)
        assert not np.all(result[195:200, 200:300] == 0)

    def test_padding_is_clamped_at_frame_edges(self, frame):
        anonymizer = Anonymizer(AnonymizationConfig(method="mask", box_padding=0.5))

        result = anonymizer.anonymize_frame(frame, [detection(0, 0, 40, 40)])

        assert result.shape == frame.shape


class TestMaskColour:
    def test_mask_colour_is_interpreted_as_rgb(self, frame):
        """Config documents RGB; frames are BGR, so the channels must swap."""
        anonymizer = Anonymizer(
            AnonymizationConfig(method="mask", mask_color=(255, 0, 0), box_padding=0.0)
        )

        result = anonymizer.anonymize_frame(frame, [detection(100, 100, 200, 200)])

        pixel = result[150, 150]
        assert tuple(pixel) == (0, 0, 255), "red RGB should land in the BGR red channel"


class TestNoAliasing:
    def test_anonymize_frame_does_not_mutate_its_input(self, frame):
        anonymizer = Anonymizer(AnonymizationConfig(method="blur"))
        before = frame.copy()

        anonymizer.anonymize_frame(frame, [detection(100, 100, 200, 200)])

        assert np.array_equal(frame, before)

    def test_every_detection_is_applied(self, frame):
        """The single-copy rewrite must not drop detections."""
        anonymizer = Anonymizer(AnonymizationConfig(method="mask", box_padding=0.0))
        boxes = [detection(x, 50, x + 40, 90) for x in (10, 100, 200, 300)]

        result = anonymizer.anonymize_frame(frame, boxes)

        for box in boxes:
            assert np.all(result[50:90, box.x1:box.x2] == 0)
