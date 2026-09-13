"""Video utility functions."""

import cv2
import numpy as np
from typing import Tuple
from loguru import logger


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target size.

    Args:
        frame: Input frame
        target_size: Target (width, height)

    Returns:
        Resized frame
    """
    if frame.shape[:2][::-1] == target_size:
        return frame

    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def validate_frame(frame: np.ndarray) -> bool:
    """Validate frame format and content.

    Args:
        frame: Frame to validate

    Returns:
        True if frame is valid
    """
    if not isinstance(frame, np.ndarray):
        logger.error("Frame is not a numpy array")
        return False

    if len(frame.shape) != 3:
        logger.error(f"Frame has invalid shape: {frame.shape}")
        return False

    if frame.shape[2] not in [3, 4]:  # BGR or BGRA
        logger.error(f"Frame has invalid channel count: {frame.shape[2]}")
        return False

    if frame.dtype != np.uint8:
        logger.warning(f"Frame has non-uint8 dtype: {frame.dtype}")
        return False

    return True


def get_frame_info(frame: np.ndarray) -> dict:
    """Get information about a frame.

    Args:
        frame: Input frame

    Returns:
        Dictionary with frame information
    """
    return {
        "shape": frame.shape,
        "dtype": str(frame.dtype),
        "size_bytes": frame.nbytes,
        "height": frame.shape[0],
        "width": frame.shape[1],
        "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
    }
