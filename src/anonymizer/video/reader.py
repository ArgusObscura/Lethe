"""Video reading and frame extraction."""

import cv2
from pathlib import Path
from typing import Optional, Generator, Tuple
import numpy as np
from loguru import logger


class VideoReader:
    """Handle video input and frame extraction."""

    def __init__(self, video_path: str):
        """Initialize video reader.

        Args:
            video_path: Path to input video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Extract video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        logger.info(f"Opened video: {self.video_path.name}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Total frames: {self.frame_count}")
        logger.info(f"  Duration: {self.duration_seconds:.2f}s")

    @property
    def duration_seconds(self) -> float:
        """Get video duration in seconds."""
        if self.fps == 0:
            return 0
        return self.frame_count / self.fps

    @property
    def codec_fourcc(self) -> str:
        """Get codec as 4-character code string."""
        codec_int = int(self.codec)
        return "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame by number.

        Args:
            frame_number: Frame index (0-based)

        Returns:
            Frame as numpy array (BGR), or None if failed
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        return frame if ret else None

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate through all frames in video.

        Yields:
            Tuple of (frame_number, frame_array)
        """
        frame_number = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            yield frame_number, frame
            frame_number += 1

    def read_frame_batch(self, batch_size: int = 8) -> Generator[list, None, None]:
        """Read frames in batches.

        Args:
            batch_size: Number of frames per batch

        Yields:
            List of (frame_number, frame_array) tuples
        """
        batch = []
        frame_number = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                if batch:  # Yield remaining frames
                    yield batch
                break

            batch.append((frame_number, frame))
            frame_number += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

    def reset(self):
        """Reset to beginning of video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            logger.info(f"Closed video: {self.video_path.name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
