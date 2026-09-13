"""Anonymization techniques for detected objects."""

import cv2
import numpy as np
from typing import List, Tuple
from loguru import logger

from .detector import Detection
from .models.config import AnonymizationMethod, AnonymizationConfig


class Anonymizer:
    """Apply anonymization techniques to detected objects."""

    def __init__(self, config: AnonymizationConfig):
        """Initialize anonymizer.

        Args:
            config: Anonymization configuration
        """
        self.config = config
        logger.info(f"Anonymizer initialized with method: {config.method}")

    def anonymize_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        method: str = None,
    ) -> np.ndarray:
        """Apply anonymization to all detections in frame.

        Args:
            frame: Input frame
            detections: List of detections
            method: Anonymization method (uses config default if None)

        Returns:
            Anonymized frame
        """
        result = frame.copy()

        if method is None:
            method = self.config.method

        for detection in detections:
            result = self.anonymize_detection(result, detection, method)

        return result

    def anonymize_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        method: str = None,
    ) -> np.ndarray:
        """Apply anonymization to a single detection.

        Args:
            frame: Input frame
            detection: Detection to anonymize
            method: Anonymization method

        Returns:
            Frame with anonymized detection
        """
        if method is None:
            method = self.config.method

        x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)

        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if method == AnonymizationMethod.BLUR or method == "blur":
            return self.blur_region(frame, x1, y1, x2, y2)
        elif method == AnonymizationMethod.PIXELATE or method == "pixelate":
            return self.pixelate_region(frame, x1, y1, x2, y2)
        elif method == AnonymizationMethod.MASK or method == "mask":
            return self.mask_region(frame, x1, y1, x2, y2)
        else:
            logger.warning(f"Unknown anonymization method: {method}, using blur")
            return self.blur_region(frame, x1, y1, x2, y2)

    def blur_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        kernel_size: int = None,
    ) -> np.ndarray:
        """Apply Gaussian blur to region.

        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
            kernel_size: Blur kernel size (uses config if None)

        Returns:
            Frame with blurred region
        """
        if kernel_size is None:
            kernel_size = self.config.blur_kernel_size

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        result = frame.copy()
        roi = result[y1:y2, x1:x2]

        if roi.size == 0:
            return result

        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        result[y1:y2, x1:x2] = blurred

        return result

    def pixelate_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        block_size: int = None,
    ) -> np.ndarray:
        """Apply pixelation to region.

        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
            block_size: Size of pixel blocks (uses config if None)

        Returns:
            Frame with pixelated region
        """
        if block_size is None:
            block_size = self.config.pixelate_size

        result = frame.copy()
        roi = result[y1:y2, x1:x2]

        if roi.size == 0:
            return result

        # Downsample
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)

        # Upsample back to original size
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y1:y2, x1:x2] = pixelated

        return result

    def mask_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int] = None,
    ) -> np.ndarray:
        """Replace region with solid color.

        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
            color: RGB color for mask (uses config if None)

        Returns:
            Frame with masked region
        """
        if color is None:
            color = self.config.mask_color

        result = frame.copy()
        result[y1:y2, x1:x2] = color

        return result

    def blur_multiple_regions(
        self,
        frame: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        kernel_size: int = None,
    ) -> np.ndarray:
        """Apply blur to multiple regions efficiently.

        Args:
            frame: Input frame
            regions: List of (x1, y1, x2, y2) coordinates
            kernel_size: Blur kernel size

        Returns:
            Frame with all regions blurred
        """
        result = frame.copy()

        for x1, y1, x2, y2 in regions:
            result = self.blur_region(result, x1, y1, x2, y2, kernel_size)

        return result

    def draw_boxes(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes on frame (for debugging).

        Args:
            frame: Input frame
            detections: List of detections
            color: Box color (BGR)
            thickness: Line thickness

        Returns:
            Frame with drawn boxes
        """
        result = frame.copy()

        for detection in detections:
            x1, y1 = int(detection.x1), int(detection.y1)
            x2, y2 = int(detection.x2), int(detection.y2)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return result
