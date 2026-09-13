"""Object detection module for faces and license plates."""

from typing import List, Tuple, Optional, Dict
import numpy as np
from loguru import logger

from .models.loader import ModelLoader
from .models.config import DetectionConfig, ObjectType


class Detection:
    """Represents a single detection result."""

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        class_name: str,
    ):
        """Initialize detection.

        Args:
            x1, y1, x2, y2: Bounding box coordinates (top-left, bottom-right)
            confidence: Detection confidence (0-1)
            class_name: Class name (face, license_plate)
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_name = class_name

    @property
    def width(self) -> float:
        """Box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Box center (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class": self.class_name,
        }

    def __repr__(self) -> str:
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, box=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}))"


class ObjectDetector:
    """Detect faces and license plates in images."""

    def __init__(self, model_loader: ModelLoader, config: DetectionConfig):
        """Initialize detector.

        Args:
            model_loader: ModelLoader instance
            config: Detection configuration
        """
        self.loader = model_loader
        self.config = config

        self.face_model = None
        self.lp_model = None

        logger.info("ObjectDetector initialized")

    def detect_faces(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of Detection objects
        """
        if self.face_model is None:
            self.face_model = self.loader.load_face_detector(self.config.device)

        try:
            results = self.face_model(frame, conf=self.config.confidence_threshold)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    detection = Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=confidence,
                        class_name="face",
                    )
                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} faces")
            return detections

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def detect_license_plates(self, frame: np.ndarray) -> List[Detection]:
        """Detect license plates in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of Detection objects
        """
        if self.lp_model is None:
            self.lp_model = self.loader.load_license_plate_detector(self.config.device)

        try:
            results = self.lp_model(frame, conf=self.config.confidence_threshold)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    detection = Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=confidence,
                        class_name="license_plate",
                    )
                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} license plates")
            return detections

        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return []

    def detect_all(self, frame: np.ndarray) -> Dict[str, List[Detection]]:
        """Detect all objects (faces and license plates) in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with detections by class
        """
        results = {}

        results["faces"] = self.detect_faces(frame)
        results["license_plates"] = self.detect_license_plates(frame)

        return results

    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: float = 0.5,
        min_area: int = 0,
    ) -> List[Detection]:
        """Filter detections by confidence and size.

        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area

        Returns:
            Filtered detections
        """
        filtered = []
        for detection in detections:
            if detection.confidence >= min_confidence and detection.area >= min_area:
                filtered.append(detection)

        return filtered

    def get_detections_by_class(
        self, detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """Group detections by class.

        Args:
            detections: List of detections

        Returns:
            Dictionary of detections grouped by class
        """
        grouped = {}
        for detection in detections:
            if detection.class_name not in grouped:
                grouped[detection.class_name] = []
            grouped[detection.class_name].append(detection)

        return grouped
