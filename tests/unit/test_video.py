"""Unit tests for video input/output."""

import pytest
import cv2
import numpy as np
from pathlib import Path

from anonymizer.video.reader import VideoReader
from anonymizer.video.writer import VideoWriter
from anonymizer.video.utils import resize_frame, validate_frame, get_frame_info


class TestVideoReader:
    """Test VideoReader class."""

    def test_reader_init(self, create_test_video):
        """Test video reader initialization."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)

        reader = VideoReader(str(video_path))

        assert reader.fps == 30
        assert reader.frame_count == 30
        assert reader.width == 640
        assert reader.height == 480

        reader.close()

    def test_reader_file_not_found(self):
        """Test reader with non-existent file."""
        with pytest.raises(FileNotFoundError):
            VideoReader("nonexistent_video.mp4")

    def test_reader_duration(self, create_test_video):
        """Test duration calculation."""
        video_path = create_test_video("test.mp4", fps=30, duration=2)

        reader = VideoReader(str(video_path))

        assert abs(reader.duration_seconds - 2.0) < 0.1

        reader.close()

    def test_reader_codec_fourcc(self, create_test_video):
        """Test codec detection."""
        video_path = create_test_video("test.mp4")

        reader = VideoReader(str(video_path))

        # Should have a codec
        assert reader.codec_fourcc

        reader.close()

    def test_reader_read_frames(self, create_test_video):
        """Test reading all frames."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)

        reader = VideoReader(str(video_path))

        frames = list(reader.read_frames())

        assert len(frames) == 30
        assert frames[0][0] == 0  # First frame number
        assert frames[-1][0] == 29  # Last frame number

        # Check frame shapes
        for frame_num, frame in frames:
            assert frame.shape == (480, 640, 3)

        reader.close()

    def test_reader_get_frame(self, create_test_video):
        """Test getting specific frame."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)

        reader = VideoReader(str(video_path))

        frame = reader.get_frame(15)

        assert frame is not None
        assert frame.shape == (480, 640, 3)

        reader.close()

    def test_reader_get_frame_out_of_bounds(self, create_test_video):
        """Test getting frame out of bounds."""
        video_path = create_test_video("test.mp4")

        reader = VideoReader(str(video_path))

        frame = reader.get_frame(1000)

        assert frame is None

        reader.close()

    def test_reader_batch_reading(self, create_test_video):
        """Test batch frame reading."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)

        reader = VideoReader(str(video_path))

        batches = list(reader.read_frame_batch(batch_size=10))

        assert len(batches) == 3  # 30 frames / 10 batch size
        assert len(batches[0]) == 10
        assert len(batches[2]) == 10

        reader.close()

    def test_reader_context_manager(self, create_test_video):
        """Test using reader as context manager."""
        video_path = create_test_video("test.mp4")

        with VideoReader(str(video_path)) as reader:
            assert reader.frame_count == 30


class TestVideoWriter:
    """Test VideoWriter class."""

    def test_writer_init(self, tmp_path):
        """Test video writer initialization."""
        output_path = tmp_path / "output.mp4"

        writer = VideoWriter(str(output_path), fps=30, width=640, height=480)

        assert writer.fps == 30
        assert writer.width == 640
        assert writer.height == 480

        writer.close()

    def test_writer_write_frame(self, tmp_path):
        """Test writing frames."""
        output_path = tmp_path / "output.mp4"

        writer = VideoWriter(str(output_path), fps=30, width=640, height=480)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        success = writer.write_frame(frame)

        assert success

        writer.close()

    def test_writer_multiple_frames(self, tmp_path):
        """Test writing multiple frames."""
        output_path = tmp_path / "output.mp4"

        writer = VideoWriter(str(output_path), fps=30, width=640, height=480)

        # Write 30 frames
        for i in range(30):
            frame = np.full((480, 640, 3), i, dtype=np.uint8)
            assert writer.write_frame(frame)

        writer.close()

        # Verify file was created
        assert output_path.exists()

    def test_writer_context_manager(self, tmp_path):
        """Test using writer as context manager."""
        output_path = tmp_path / "output.mp4"

        with VideoWriter(str(output_path), fps=30, width=640, height=480) as writer:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write_frame(frame)

        assert output_path.exists()


class TestVideoUtils:
    """Test video utility functions."""

    def test_resize_frame(self):
        """Test frame resizing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_frame(frame, (320, 240))

        assert resized.shape == (240, 320, 3)

    def test_resize_same_size(self):
        """Test resizing to same size."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_frame(frame, (640, 480))

        # Should return original
        assert np.array_equal(resized, frame)

    def test_validate_frame_valid(self):
        """Test frame validation with valid frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        assert validate_frame(frame)

    def test_validate_frame_invalid_type(self):
        """Test frame validation with invalid type."""
        frame = []  # Not an array

        assert not validate_frame(frame)

    def test_validate_frame_invalid_shape(self):
        """Test frame validation with invalid shape."""
        frame = np.zeros((480, 640), dtype=np.uint8)  # 2D instead of 3D

        assert not validate_frame(frame)

    def test_validate_frame_invalid_channels(self):
        """Test frame validation with invalid channels."""
        frame = np.zeros((480, 640, 5), dtype=np.uint8)  # 5 channels

        assert not validate_frame(frame)

    def test_get_frame_info(self):
        """Test getting frame information."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        info = get_frame_info(frame)

        assert info["height"] == 480
        assert info["width"] == 640
        assert info["channels"] == 3
        assert info["dtype"] == "uint8"
