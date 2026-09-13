"""Frame stream input for live and networked sources."""

import re
import time
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger


# Webcams and some network sources report a nonsense FPS; fall back to this
# so the output file plays back at a sane rate.
DEFAULT_FPS = 30.0

_WEBCAM_SCHEME = re.compile(r"^webcam://(\d+)$", re.IGNORECASE)


def parse_source(source: str) -> Union[int, str]:
    """Translate a user-facing source string into a cv2.VideoCapture argument.

    Accepts ``webcam://0``, a bare device index, or any URL/path OpenCV
    understands (``rtsp://``, ``http://``, a file path).
    """
    match = _WEBCAM_SCHEME.match(source)
    if match:
        return int(match.group(1))

    if source.isdigit():
        return int(source)

    return source


class FrameStream:
    """A frame source that may be live and unbounded.

    Unlike :class:`~anonymizer.video.reader.VideoReader`, a stream has no
    reliable frame count and can end at any time, so callers bound it with
    ``max_frames`` or ``max_seconds`` rather than iterating to completion.
    """

    def __init__(self, source: str, read_timeout: float = 10.0):
        """Open a stream.

        Args:
            source: ``webcam://0``, an RTSP/HTTP URL, or a file path.
            read_timeout: Seconds of consecutive read failures to tolerate
                before giving up. Network streams drop frames routinely.

        Raises:
            ValueError: If the source cannot be opened.
        """
        self.source = source
        self.read_timeout = read_timeout

        capture_arg = parse_source(source)
        self.cap = cv2.VideoCapture(capture_arg)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open stream: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps and reported_fps > 0 else DEFAULT_FPS

        # Negative or zero for live sources, which have no end.
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count: Optional[int] = count if count > 0 else None

        logger.info(f"Opened stream: {source}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}{'' if reported_fps > 0 else ' (assumed)'}")
        logger.info(f"  Frame count: {self.frame_count or 'live (unbounded)'}")

    @property
    def is_live(self) -> bool:
        """Whether the stream has no known end."""
        return self.frame_count is None

    def read_frames(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield ``(frame_number, frame)`` until a limit or the stream ends.

        Args:
            max_frames: Stop after this many frames.
            max_seconds: Stop after this much wall-clock time.
        """
        frame_number = 0
        started = time.monotonic()
        last_good_read = started

        while True:
            if max_frames is not None and frame_number >= max_frames:
                logger.info(f"Reached max_frames limit ({max_frames})")
                break

            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                logger.info(f"Reached max_seconds limit ({max_seconds}s)")
                break

            ret, frame = self.cap.read()

            if not ret:
                # A file-backed source is simply finished. A live one may just
                # be starved, so keep trying until the timeout expires.
                if not self.is_live:
                    break

                if time.monotonic() - last_good_read >= self.read_timeout:
                    logger.warning(
                        f"No frames for {self.read_timeout}s, ending stream"
                    )
                    break

                time.sleep(0.01)
                continue

            last_good_read = time.monotonic()
            yield frame_number, frame
            frame_number += 1

    def close(self):
        """Release the underlying capture."""
        if self.cap:
            self.cap.release()
            logger.info(f"Closed stream: {self.source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
