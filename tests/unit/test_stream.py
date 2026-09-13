"""Tests for frame stream input."""

import cv2
import numpy as np
import pytest

from anonymizer.video.stream import DEFAULT_FPS, FrameStream, parse_source


@pytest.fixture
def sample_video(tmp_path):
    """A short synthetic video on disk."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


class TestParseSource:
    def test_webcam_scheme_yields_index(self):
        assert parse_source("webcam://0") == 0
        assert parse_source("webcam://2") == 2

    def test_webcam_scheme_is_case_insensitive(self):
        assert parse_source("WEBCAM://1") == 1

    def test_bare_digit_yields_index(self):
        assert parse_source("0") == 0

    def test_urls_pass_through(self):
        assert parse_source("rtsp://cam.local/s") == "rtsp://cam.local/s"
        assert parse_source("http://host:8000/mjpeg") == "http://host:8000/mjpeg"

    def test_file_path_passes_through(self):
        assert parse_source("/videos/clip.mp4") == "/videos/clip.mp4"


class TestFrameStream:
    def test_rejects_unopenable_source(self):
        with pytest.raises(ValueError, match="Failed to open stream"):
            FrameStream("rtsp://127.0.0.1:1/nonexistent")

    def test_reads_file_backed_stream_to_completion(self, sample_video):
        with FrameStream(sample_video) as stream:
            frames = list(stream.read_frames())

        assert len(frames) == 10
        assert [n for n, _ in frames] == list(range(10))

    def test_file_source_is_not_live(self, sample_video):
        with FrameStream(sample_video) as stream:
            assert stream.is_live is False
            assert stream.frame_count == 10

    def test_reports_dimensions(self, sample_video):
        with FrameStream(sample_video) as stream:
            assert (stream.width, stream.height) == (64, 48)

    def test_max_frames_bounds_the_stream(self, sample_video):
        with FrameStream(sample_video) as stream:
            frames = list(stream.read_frames(max_frames=4))

        assert len(frames) == 4

    def test_max_seconds_bounds_the_stream(self, sample_video):
        with FrameStream(sample_video) as stream:
            frames = list(stream.read_frames(max_seconds=0))

        assert frames == []

    def test_falls_back_to_default_fps_when_unreported(self, sample_video, monkeypatch):
        real_get = cv2.VideoCapture.get

        def fake_get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return real_get(self, prop)

        monkeypatch.setattr(cv2.VideoCapture, "get", fake_get)

        with FrameStream(sample_video) as stream:
            assert stream.fps == DEFAULT_FPS

    def test_close_is_idempotent(self, sample_video):
        stream = FrameStream(sample_video)
        stream.close()
        stream.close()
