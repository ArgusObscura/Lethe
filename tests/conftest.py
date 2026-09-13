"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def test_video_dir(tmp_path):
    """Create a temporary directory for test videos."""
    return tmp_path / "videos"


@pytest.fixture
def test_frame_dir(tmp_path):
    """Create a temporary directory for test frames."""
    return tmp_path / "frames"


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    # Create a 480x640 RGB frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some content
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(frame, (500, 200), 50, (0, 0, 255), -1)  # Red circle

    return frame


@pytest.fixture
def create_test_video(tmp_path):
    """Factory fixture to create test videos."""
    def _create_video(filename, fps=30, duration=2, width=640, height=480):
        """Create a simple test video."""
        video_path = tmp_path / filename

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Write frames
        num_frames = int(fps * duration)
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add some motion
            x = int(100 + 50 * np.sin(i / 10.0))
            y = int(100 + 50 * np.cos(i / 10.0))
            cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)

            # Add text
            cv2.putText(
                frame,
                f"Frame {i+1}/{num_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            out.write(frame)

        out.release()
        return video_path

    return _create_video


@pytest.fixture
def create_test_image(tmp_path):
    """Factory fixture to create test images."""
    def _create_image(filename, width=640, height=480, color=(0, 255, 0)):
        """Create a simple test image."""
        image_path = tmp_path / filename

        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add content
        cv2.rectangle(image, (50, 50), (300, 300), color, -1)
        cv2.circle(image, (500, 200), 50, (0, 0, 255), -1)
        cv2.putText(
            image,
            "Test Image",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imwrite(str(image_path), image)
        return image_path

    return _create_image
