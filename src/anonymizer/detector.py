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

    def __init__(
        self,
        model_loader: ModelLoader,
        config: DetectionConfig,
        license_plate_config: Optional[DetectionConfig] = None,
    ):
        """Initialize detector.

        Args:
            model_loader: ModelLoader instance
            config: Detection configuration for faces
            license_plate_config: Detection configuration for license plates.
                Falls back to ``config`` when not given.
        """
        self.loader = model_loader
        self.config = config
        self.license_plate_config = license_plate_config or config

        self.face_model = None
        self.lp_model = None
        self.person_model = None

        # People are far larger than faces, so confirming one needs much less
        # resolution than finding a face does.
        self.person_confidence = 0.3
        self.person_inference_size = 960

        logger.info("ObjectDetector initialized")

    @staticmethod
    def gate_faces_by_person(
        faces: List[Detection],
        persons: List[Detection],
        min_size: int,
        min_overlap: float = 0.6,
    ) -> List[Detection]:
        """Drop large face detections that do not sit on a detected person.

        The face model reads car rear ends, wheels and windscreen reflections
        as faces — two lights and a plate make a convincing arrangement — and
        does so confidently, so a confidence threshold cannot separate them.
        Those false positives are large; real distant faces are not.

        Small detections pass through ungated: the person detector is
        unreliable at that distance, and a spurious few-pixel blur costs far
        less than dropping a real face.

        Args:
            faces: Face detections for one frame.
            persons: Person detections for the same frame.
            min_size: Width in pixels at or above which person support is
                required.
            min_overlap: Fraction of the face box that must fall inside a
                person box.

        Returns:
            The face detections that survive gating.
        """
        def supported(face: Detection) -> bool:
            area = (face.x2 - face.x1) * (face.y2 - face.y1)
            if area <= 0:
                return False

            for person in persons:
                ix1, iy1 = max(face.x1, person.x1), max(face.y1, person.y1)
                ix2, iy2 = min(face.x2, person.x2), min(face.y2, person.y2)
                overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if overlap / area >= min_overlap:
                    return True

            return False

        return [
            face
            for face in faces
            if (face.x2 - face.x1) < min_size or supported(face)
        ]

    @staticmethod
    def _size_kwargs(config, frames=None) -> dict:
        """Inference size for a model call.

        An explicit ``inference_size`` wins. Otherwise it follows the footage's
        own width, because a detector's usual 640 default silently downscales
        anything larger — on 1280px dashcam footage that halves every face, and
        on 2730px footage it shrinks a 50px face to 12px.
        """
        if config.inference_size:
            return {"imgsz": config.inference_size}

        if not frames:
            return {}

        width = frames[0].shape[1] if hasattr(frames[0], "shape") else None
        if not width:
            return {}

        return {"imgsz": ObjectDetector.auto_inference_size(width)}

    # Above this, the cost stops buying enough recall to be worth paying by
    # default; raise inference_size explicitly for very high resolution work.
    MAX_AUTO_INFERENCE_SIZE = 1920
    MIN_AUTO_INFERENCE_SIZE = 640

    @staticmethod
    def auto_inference_size(frame_width: int) -> int:
        """Inference size to use for footage of this width.

        Rounded up to a multiple of 32, which is the stride YOLO expects.
        """
        size = max(
            ObjectDetector.MIN_AUTO_INFERENCE_SIZE,
            min(int(frame_width), ObjectDetector.MAX_AUTO_INFERENCE_SIZE),
        )
        return ((size + 31) // 32) * 32

    @staticmethod
    def _to_detections(result, class_name: str) -> List[Detection]:
        """Convert one YOLO result into Detection objects."""
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=float(box.conf[0].cpu().numpy()),
                    class_name=class_name,
                )
            )

        return detections

    def _load_face_model(self):
        if self.face_model is None:
            self.face_model = self.loader.load_face_detector(self.config.device)
        return self.face_model

    def _load_lp_model(self):
        if self.lp_model is None:
            self.lp_model = self.loader.load_license_plate_detector(
                self.license_plate_config.device
            )
        return self.lp_model

    def detect_faces(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of Detection objects
        """
        model = self._load_face_model()

        try:
            results = model(frame, conf=self.config.confidence_threshold, **self._size_kwargs(self.config, [frame]))
            detections = [
                d for result in results for d in self._to_detections(result, "face")
            ]

            logger.debug(f"Detected {len(detections)} faces")
            return detections

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def detect_faces_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect faces across several frames in one forward pass.

        Args:
            frames: Input frames (BGR)

        Returns:
            One list of Detections per input frame, in the same order.
        """
        model = self._load_face_model()

        try:
            results = model(frames, conf=self.config.confidence_threshold, **self._size_kwargs(self.config, frames))
            return [self._to_detections(result, "face") for result in results]

        except Exception as e:
            logger.error(f"Batched face detection failed: {e}")
            return [[] for _ in frames]

    def detect_persons_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect people across several frames.

        Used to confirm that a face detection sits on a person. People are far
        larger than faces, so this runs at a lower resolution than the face
        pass.

        Args:
            frames: Input frames (BGR)

        Returns:
            One list of person Detections per input frame.
        """
        if self.person_model is None:
            self.person_model = self.loader.load_person_detector(self.config.device)

        try:
            results = self.person_model(
                frames,
                conf=self.person_confidence,
                imgsz=self.person_inference_size,
            )
            return [self._persons_only(result) for result in results]
        except Exception as e:
            logger.error(f"Person detection failed: {e}")
            return [[] for _ in frames]

    @staticmethod
    def _persons_only(result) -> List[Detection]:
        """Keep just the person class out of a COCO result."""
        detections = []

        for box in result.boxes:
            if result.names[int(box.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=float(box.conf[0].cpu().numpy()),
                    class_name="person",
                )
            )

        return detections

    def detect_license_plates_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Detection]]:
        """Detect license plates across several frames in one forward pass.

        Args:
            frames: Input frames (BGR)

        Returns:
            One list of Detections per input frame, in the same order.
        """
        model = self._load_lp_model()

        try:
            results = model(
                frames,
                conf=self.license_plate_config.confidence_threshold,
                **self._size_kwargs(self.license_plate_config, frames),
            )
            return [
                self._to_detections(result, "license_plate") for result in results
            ]

        except Exception as e:
            logger.error(f"Batched license plate detection failed: {e}")
            return [[] for _ in frames]

    def detect_license_plates(self, frame: np.ndarray) -> List[Detection]:
        """Detect license plates in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of Detection objects
        """
        model = self._load_lp_model()

        try:
            results = model(
                frame,
                conf=self.license_plate_config.confidence_threshold,
                **self._size_kwargs(self.license_plate_config, [frame]),
            )
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
