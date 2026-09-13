"""Model loading and caching for Lethe."""

import os
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO
from loguru import logger


class ModelLoader:
    """Load and cache detection models."""

    # Default model paths
    YOLOV8_FACE_MODEL = "yolov8n-face.pt"
    YOLOV8_LP_MODEL = "yolov8m.pt"  # Generic YOLOv8, fine-tune for LP detection

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models.
                      Defaults to ~/.cache/lethe/models
        """
        if cache_dir is None:
            cache_dir = str(Path.home() / ".cache" / "lethe" / "models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, YOLO] = {}
        logger.info(f"Model cache directory: {self.cache_dir}")

    def load_face_detector(self, device: str = "cpu") -> YOLO:
        """Load YOLOv8-face model for face detection.

        Args:
            device: Device to load model on ('cpu' or 'cuda')

        Returns:
            YOLO model for face detection
        """
        model_name = "face_detector"

        if model_name in self._models:
            return self._models[model_name]

        logger.info("Loading YOLOv8-face model for face detection...")
        try:
            # Use cache directory and set YOLO to download there
            os.environ['YOLO_CACHE'] = str(self.cache_dir)
            model = YOLO(self.YOLOV8_FACE_MODEL)
            model.to(device)
            self._models[model_name] = model
            logger.info(f"Face detector loaded successfully from {self.cache_dir}")
            return model
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            logger.info("Trying to download model from Ultralytics hub...")
            try:
                # Try downloading from Ultralytics hub
                os.environ['YOLO_CACHE'] = str(self.cache_dir)
                model = YOLO(self.YOLOV8_FACE_MODEL)
                model.to(device)
                self._models[model_name] = model
                logger.info("Face detector downloaded and loaded successfully")
                return model
            except Exception as e2:
                logger.error(f"Failed to download face detector: {e2}")
                raise

    def load_license_plate_detector(self, device: str = "cpu") -> YOLO:
        """Load YOLOv8 model for license plate detection.

        Args:
            device: Device to load model on ('cpu' or 'cuda')

        Returns:
            YOLO model for license plate detection
        """
        model_name = "lp_detector"

        if model_name in self._models:
            return self._models[model_name]

        logger.info("Loading YOLOv8 model for license plate detection...")
        try:
            # Use cache directory and set YOLO to download there
            os.environ['YOLO_CACHE'] = str(self.cache_dir)
            model = YOLO(self.YOLOV8_LP_MODEL)
            model.to(device)
            self._models[model_name] = model
            logger.info(f"License plate detector loaded successfully from {self.cache_dir}")
            return model
        except Exception as e:
            logger.error(f"Failed to load license plate detector: {e}")
            raise

    def load_custom_model(self, model_path: str, device: str = "cpu") -> YOLO:
        """Load a custom YOLO model.

        Args:
            model_path: Path to custom model file
            device: Device to load model on

        Returns:
            Loaded YOLO model
        """
        logger.info(f"Loading custom model from {model_path}...")
        try:
            model = YOLO(model_path)
            model.to(device)
            logger.info("Custom model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise

    def unload_all(self):
        """Unload all cached models to free memory."""
        self._models.clear()
        logger.info("All models unloaded")

    def get_cached_models(self) -> Dict[str, YOLO]:
        """Get dictionary of all cached models."""
        return self._models.copy()
