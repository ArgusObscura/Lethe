"""Anonymization techniques for detected objects."""

import cv2
import numpy as np
from typing import List, Tuple
from loguru import logger

from .detector import Detection
from .models.config import AnonymizationMethod, AnonymizationConfig


class Anonymizer:
    """Apply anonymization techniques to detected objects."""

    # Upper bound on pixelation blocks across a region, so anonymization
    # strength does not weaken as a face grows on screen.
    MAX_PIXELATE_BLOCKS = 12

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
        if method is None:
            method = self.config.method

        result = frame.copy()

        for detection in detections:
            self._apply_to_detection(result, detection, method)

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

        result = frame.copy()
        self._apply_to_detection(result, detection, method)

        return result

    def _detection_box(
        self, frame: np.ndarray, detection: Detection
    ) -> Tuple[int, int, int, int]:
        """Expand a detection box and clamp it to the frame.

        Detectors return boxes drawn tightly around the feature, which leaves
        hair, jawline and ear edges outside the anonymized region. Padding the
        box covers what the detector considered outside the face.
        """
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)

        pad = self.config.box_padding
        if pad:
            pad_x = int((x2 - x1) * pad)
            pad_y = int((y2 - y1) * pad)
            x1, y1 = x1 - pad_x, y1 - pad_y
            x2, y2 = x2 + pad_x, y2 + pad_y

        return (
            max(0, x1),
            max(0, y1),
            min(frame.shape[1], x2),
            min(frame.shape[0], y2),
        )

    def _apply_to_detection(
        self, frame: np.ndarray, detection: Detection, method: str
    ) -> None:
        """Anonymize one detection, modifying ``frame`` in place."""
        x1, y1, x2, y2 = self._detection_box(frame, detection)

        if method == AnonymizationMethod.PIXELATE or method == "pixelate":
            self._pixelate_in_place(frame, x1, y1, x2, y2, self.config.pixelate_size)
        elif method == AnonymizationMethod.MASK or method == "mask":
            self._mask_in_place(frame, x1, y1, x2, y2, self.config.mask_color)
        else:
            if not (method == AnonymizationMethod.BLUR or method == "blur"):
                logger.warning(f"Unknown anonymization method: {method}, using blur")
            self._blur_in_place(frame, x1, y1, x2, y2, self.config.blur_kernel_size)

    def _blur_in_place(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, kernel_size: int
    ) -> None:
        """Blur a region with strength proportional to its size.

        A fixed kernel anonymizes weakly exactly where it matters most: a face
        filling the frame keeps far more identifying structure than a distant
        one, because the blur radius is unrelated to the feature scale. Sigma
        therefore scales with the box, using the configured kernel as a floor.
        """
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return

        height, width = roi.shape[:2]
        config_sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        sigma = max(config_sigma, max(width, height) / 6.0)

        # ksize (0, 0) lets OpenCV derive the kernel from sigma.
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), sigma)

    def _pixelate_in_place(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, block_size: int
    ) -> None:
        """Pixelate a region, modifying ``frame`` in place.

        A fixed pixel block size has the same flaw as a fixed blur kernel: a
        500px face keeps 33 blocks across and stays identifiable, while a 20px
        one collapses to a single block. Capping the block count makes the
        result equally coarse whatever the face's size on screen.
        """
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return

        height, width = roi.shape[:2]

        # A region smaller than one block still collapses to a single block;
        # without the floor the intermediate resize gets a zero dimension.
        blocks_x = max(1, min(width // block_size, self.MAX_PIXELATE_BLOCKS))
        blocks_y = max(1, min(height // block_size, self.MAX_PIXELATE_BLOCKS))

        small = cv2.resize(
            roi, (blocks_x, blocks_y), interpolation=cv2.INTER_LINEAR
        )
        frame[y1:y2, x1:x2] = cv2.resize(
            small, (width, height), interpolation=cv2.INTER_NEAREST
        )

    def _mask_in_place(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Fill a region with a solid colour, modifying ``frame`` in place."""
        # Config expresses the colour as RGB; frames are BGR.
        frame[y1:y2, x1:x2] = tuple(reversed(tuple(color)))

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

        result = frame.copy()
        self._blur_in_place(result, x1, y1, x2, y2, kernel_size)

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
        self._pixelate_in_place(result, x1, y1, x2, y2, block_size)

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
        self._mask_in_place(result, x1, y1, x2, y2, color)

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
