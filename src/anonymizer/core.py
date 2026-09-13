"""Core anonymization pipeline."""

from pathlib import Path
from typing import Optional, Callable, Dict
from tqdm import tqdm
from loguru import logger

from .video.reader import VideoReader
from .video.writer import VideoWriter, FFmpegVideoWriter
from .detector import ObjectDetector, Detection
from .anonymizer import Anonymizer
from .models.loader import ModelLoader
from .models.config import AnonymizationConfig, DetectionConfig


class AnonymizationPipeline:
    """Main pipeline for video anonymization."""

    def __init__(
        self,
        config: Optional[AnonymizationConfig] = None,
        use_ffmpeg: bool = False,
    ):
        """Initialize anonymization pipeline.

        Args:
            config: Anonymization configuration (uses defaults if None)
            use_ffmpeg: Use FFmpeg writer instead of OpenCV
        """
        self.config = config or AnonymizationConfig()
        self.use_ffmpeg = use_ffmpeg

        self.model_loader = ModelLoader()
        self.detector = ObjectDetector(self.model_loader, self.config.face_config)
        self.anonymizer = Anonymizer(self.config)

        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "license_plates_detected": 0,
            "total_detections": 0,
        }

        logger.info("AnonymizationPipeline initialized")

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        """Process entire video from start to finish.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            progress_callback: Callback function(current_frame, total_frames)

        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting video anonymization: {input_path} -> {output_path}")

        # Open input video
        with VideoReader(input_path) as reader:
            # Initialize output writer
            writer_class = FFmpegVideoWriter if self.use_ffmpeg else VideoWriter

            with writer_class(
                output_path,
                reader.fps,
                reader.width,
                reader.height,
                crf=self.config.face_config.batch_size if hasattr(self.config, 'crf') else 23,
            ) as writer:

                # Process frames
                pbar = tqdm(total=reader.frame_count, desc="Processing video")

                for frame_num, frame in reader.read_frames():
                    # Detect objects
                    detections = []

                    if self.config.enable_face_detection:
                        face_detections = self.detector.detect_faces(frame)
                        detections.extend(face_detections)
                        self.stats["faces_detected"] += len(face_detections)

                    if self.config.enable_license_plate_detection:
                        lp_detections = self.detector.detect_license_plates(frame)
                        detections.extend(lp_detections)
                        self.stats["license_plates_detected"] += len(lp_detections)

                    self.stats["total_detections"] += len(detections)

                    # Anonymize detections
                    anonymized_frame = self.anonymizer.anonymize_frame(frame, detections)

                    # Write frame
                    writer.write_frame(anonymized_frame)

                    # Update stats and progress
                    self.stats["frames_processed"] += 1

                    if progress_callback:
                        progress_callback(frame_num, reader.frame_count)

                    pbar.update(1)

                pbar.close()

        logger.info(f"Video processing complete. Output: {output_path}")
        logger.info(f"Statistics: {self.stats}")

        return self.stats

    def process_frame(self, frame_path: str, output_path: str) -> Dict:
        """Process a single image frame.

        Args:
            frame_path: Path to input image
            output_path: Path to output image

        Returns:
            Dictionary with detection statistics
        """
        import cv2

        logger.info(f"Processing frame: {frame_path}")

        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to load image: {frame_path}")

        # Detect objects
        detections = []

        if self.config.enable_face_detection:
            detections.extend(self.detector.detect_faces(frame))

        if self.config.enable_license_plate_detection:
            detections.extend(self.detector.detect_license_plates(frame))

        # Anonymize
        anonymized_frame = self.anonymizer.anonymize_frame(frame, detections)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, anonymized_frame)

        logger.info(f"Frame saved to: {output_path}")

        return {
            "detections": len(detections),
            "faces": sum(1 for d in detections if d.class_name == "face"),
            "license_plates": sum(1 for d in detections if d.class_name == "license_plate"),
        }

    def detect_frame(self, frame_path: str) -> Dict[str, list]:
        """Detect objects in a single frame without anonymization.

        Args:
            frame_path: Path to input image

        Returns:
            Dictionary of detections by class
        """
        import cv2

        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to load image: {frame_path}")

        detections = self.detector.detect_all(frame)

        return {
            "faces": [d.to_dict() for d in detections.get("faces", [])],
            "license_plates": [d.to_dict() for d in detections.get("license_plates", [])],
        }

    def cleanup(self):
        """Clean up resources."""
        self.model_loader.unload_all()
        logger.info("Pipeline cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
