"""Unit tests for anonymization module."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from anonymizer.anonymizer import Anonymizer
from anonymizer.detector import Detection
from anonymizer.models.config import AnonymizationConfig, AnonymizationMethod


@pytest.fixture
def test_frame():
    """Create a test frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detection():
    """Create a test detection."""
    return Detection(
        x1=100,
        y1=100,
        x2=200,
        y2=200,
        confidence=0.95,
        class_name="face",
    )


@pytest.fixture
def config():
    """Create a test config."""
    return AnonymizationConfig()


class TestAnonymizer:
    """Test Anonymizer class."""

    def test_anonymizer_init(self, config):
        """Test anonymizer initialization."""
        anonymizer = Anonymizer(config)
        assert anonymizer.config == config

    def test_blur_region(self, test_frame, config):
        """Test blur anonymization."""
        anonymizer = Anonymizer(config)

        result = anonymizer.blur_region(test_frame, 100, 100, 200, 200)

        assert result.shape == test_frame.shape
        # Blurred region should be different from original
        assert not np.array_equal(result[100:200, 100:200], test_frame[100:200, 100:200])

    def test_pixelate_region(self, test_frame, config):
        """Test pixelation anonymization."""
        anonymizer = Anonymizer(config)

        result = anonymizer.pixelate_region(test_frame, 100, 100, 200, 200)

        assert result.shape == test_frame.shape
        # Pixelated region should look blocky (not identical to original)
        assert not np.array_equal(result[100:200, 100:200], test_frame[100:200, 100:200])

    def test_mask_region(self, test_frame, config):
        """Test masking anonymization."""
        anonymizer = Anonymizer(config)

        color = (0, 0, 0)  # Black
        result = anonymizer.mask_region(test_frame, 100, 100, 200, 200, color=color)

        assert result.shape == test_frame.shape
        # Masked region should be all black
        assert np.all(result[100:200, 100:200] == 0)

    def test_anonymize_detection(self, test_frame, test_detection, config):
        """Test anonymizing a single detection."""
        anonymizer = Anonymizer(config)

        result = anonymizer.anonymize_detection(test_frame, test_detection, method="blur")

        assert result.shape == test_frame.shape
        # Region should be modified
        assert not np.array_equal(
            result[100:200, 100:200],
            test_frame[100:200, 100:200]
        )

    def test_anonymize_frame(self, test_frame, config):
        """Test anonymizing entire frame."""
        anonymizer = Anonymizer(config)

        detections = [
            Detection(50, 50, 150, 150, 0.9, "face"),
            Detection(200, 200, 300, 300, 0.85, "license_plate"),
        ]

        result = anonymizer.anonymize_frame(test_frame, detections)

        assert result.shape == test_frame.shape
        # At least one region should be modified
        assert not np.array_equal(result, test_frame)

    def test_anonymize_frame_empty_detections(self, test_frame, config):
        """Test anonymizing frame with no detections."""
        anonymizer = Anonymizer(config)

        result = anonymizer.anonymize_frame(test_frame, [])

        # Frame should be unchanged
        assert np.array_equal(result, test_frame)

    def test_draw_boxes(self, test_frame, config):
        """Test drawing detection boxes."""
        anonymizer = Anonymizer(config)

        detections = [
            Detection(50, 50, 150, 150, 0.9, "face"),
        ]

        result = anonymizer.draw_boxes(test_frame, detections)

        assert result.shape == test_frame.shape
        # Frame should be modified (boxes drawn)
        assert not np.array_equal(result, test_frame)

    def test_method_selection(self, test_frame, test_detection):
        """Test different anonymization methods."""
        config = AnonymizationConfig()
        anonymizer = Anonymizer(config)

        result_blur = anonymizer.anonymize_detection(
            test_frame.copy(),
            test_detection,
            method="blur"
        )
        result_pixelate = anonymizer.anonymize_detection(
            test_frame.copy(),
            test_detection,
            method="pixelate"
        )
        result_mask = anonymizer.anonymize_detection(
            test_frame.copy(),
            test_detection,
            method="mask"
        )

        # All results should be different from original
        assert not np.array_equal(result_blur, test_frame)
        assert not np.array_equal(result_pixelate, test_frame)
        assert not np.array_equal(result_mask, test_frame)

        # Results should be different from each other
        assert not np.array_equal(result_blur, result_pixelate)
        assert not np.array_equal(result_blur, result_mask)

    def test_boundary_cases(self, config):
        """Test anonymization at image boundaries."""
        anonymizer = Anonymizer(config)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test region at boundary
        result = anonymizer.blur_region(frame, 50, 50, 150, 150)

        assert result.shape == frame.shape
        # Should not crash and should modify the region
        assert not np.array_equal(result[50:100, 50:100], frame[50:100, 50:100])


class TestDetection:
    """Test Detection class."""

    def test_detection_creation(self):
        """Test creating a detection."""
        det = Detection(10, 20, 100, 150, 0.95, "face")

        assert det.x1 == 10
        assert det.y1 == 20
        assert det.x2 == 100
        assert det.y2 == 150
        assert det.confidence == 0.95
        assert det.class_name == "face"

    def test_detection_properties(self):
        """Test detection properties."""
        det = Detection(10, 20, 110, 220, 0.95, "face")

        assert det.width == 100
        assert det.height == 200
        assert det.area == 20000
        assert det.center == (60, 120)

    def test_detection_to_dict(self):
        """Test detection to dictionary conversion."""
        det = Detection(10, 20, 100, 150, 0.95, "face")
        d = det.to_dict()

        assert d["x1"] == 10
        assert d["y1"] == 20
        assert d["x2"] == 100
        assert d["y2"] == 150
        assert d["confidence"] == 0.95
        assert d["class"] == "face"
