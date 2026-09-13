"""Video writing and output handling."""

import cv2
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from loguru import logger


class VideoWriter:
    """Handle video output and frame writing."""

    # Codec mapping for different formats
    CODEC_MAP = {
        "mp4": "libx264",
        "avi": "mpeg4",
        "mov": "libx264",
        "mkv": "libx264",
    }

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: Optional[str] = None,
        crf: int = 23,
    ):
        """Initialize video writer using OpenCV.

        Args:
            output_path: Path for output video
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (auto-detect if None)
            crf: Quality level for H.264 (0-51, lower is better)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.width = width
        self.height = height
        self.crf = crf

        # Determine codec and fourcc
        if codec is None:
            codec = self._detect_codec_from_path()

        self.codec_name = codec
        fourcc = cv2.VideoWriter_fourcc(*self._get_fourcc(codec))

        logger.info(f"Initializing video writer for {output_path}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Codec: {codec}")

        try:
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                fps,
                (width, height)
            )

            if not self.writer.isOpened():
                raise ValueError("Failed to open video writer")

        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            raise

    def _detect_codec_from_path(self) -> str:
        """Detect codec from file extension."""
        ext = self.output_path.suffix.lower().lstrip(".")
        return self.CODEC_MAP.get(ext, "libx264")

    def _get_fourcc(self, codec: str) -> str:
        """Get FFmpeg fourcc code for codec."""
        fourcc_map = {
            "libx264": "mp4v",
            "mpeg4": "DIVX",
            "libx265": "hev1",
        }
        return fourcc_map.get(codec, "mp4v")

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a single frame to video.

        Args:
            frame: Frame as BGR numpy array

        Returns:
            True if the writer is still open afterwards, False otherwise
        """
        if frame.shape[:2] != (self.height, self.width):
            logger.warning(
                f"Frame size mismatch. Expected {self.width}x{self.height}, "
                f"got {frame.shape[1]}x{frame.shape[0]}. Resizing..."
            )
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)

        # cv2's write() returns None and reports no per-frame status, so the
        # writer staying open is the only signal available.
        return self.writer.isOpened()

    def write_frames(self, frames: list) -> bool:
        """Write multiple frames.

        Args:
            frames: List of frame arrays

        Returns:
            True if all writes successful
        """
        all_written = True
        for frame in frames:
            if not self.write_frame(frame):
                all_written = False
                logger.error("Failed to write frame")

        return all_written

    def close(self):
        """Close and finalize video file."""
        if self.writer:
            self.writer.release()
            logger.info(f"Closed video writer. Output: {self.output_path}")

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


class FFmpegVideoWriter:
    """Alternative video writer using FFmpeg for better codec support."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        crf: int = 23,
        codec: str = "libx264",
    ):
        """Initialize FFmpeg-based video writer.

        Args:
            output_path: Path for output video
            fps: Frames per second
            width: Frame width
            height: Frame height
            crf: Quality level (0-51)
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.width = width
        self.height = height
        self.crf = crf
        self.codec = codec

        logger.info(f"Initializing FFmpeg writer for {output_path}")

        # FFmpeg command
        self.ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "medium",
            str(self.output_path)
        ]

        try:
            self.process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install it or use VideoWriter instead.")
            raise

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to FFmpeg stdin."""
        try:
            self.process.stdin.write(frame.tobytes())
            return True
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False

    def close(self):
        """Close FFmpeg process."""
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            logger.info(f"Closed FFmpeg writer. Output: {self.output_path}")

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
