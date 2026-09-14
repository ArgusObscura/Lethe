"""Core anonymization pipeline."""

from pathlib import Path
from typing import Optional, Callable, Dict
from datetime import datetime
from loguru import logger

from .video.reader import VideoReader
from .video.stream import FrameStream
from .video.writer import VideoWriter, FFmpegVideoWriter
from .detector import ObjectDetector, Detection
from .anonymizer import Anonymizer
from .models.loader import ModelLoader
from .models.config import AnonymizationConfig, DetectionConfig
from .events import EventEmitter, EventType, Event, EventCallback
from .tracking import DetectionTracker


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
        self.detector = ObjectDetector(
            self.model_loader,
            self.config.face_config,
            self.config.license_plate_config,
        )
        self.anonymizer = Anonymizer(self.config)

        # A track must outlive the gap between detections, or skipping frames
        # would drop every region before the next detection refreshes it.
        persistence = max(
            self.config.track_persistence, self.config.detect_every * 3
        )

        # Faces and plates are tracked separately so a face never inherits a
        # plate's track, and vice versa.
        self.face_tracker = (
            DetectionTracker(persistence=persistence)
            if self.config.track_detections
            else None
        )
        self.plate_tracker = (
            DetectionTracker(persistence=persistence)
            if self.config.track_detections
            else None
        )

        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "license_plates_detected": 0,
            "total_detections": 0,
        }

        # Event system
        self.events = EventEmitter()

        logger.info("AnonymizationPipeline initialized")

    def on(self, event_type: EventType, callback: EventCallback) -> 'AnonymizationPipeline':
        """Register an event callback.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs

        Returns:
            Self for method chaining
        """
        self.events.on(event_type, callback)
        return self

    def _detect_faces_gated(self, frames):
        """Detect faces across a batch, discarding ones not on a person.

        Returns:
            One list of face Detections per input frame.
        """
        faces_per_frame = self.detector.detect_faces_batch(frames)

        if not self.config.person_gate:
            return faces_per_frame

        persons_per_frame = self.detector.detect_persons_batch(frames)

        return [
            self.detector.gate_faces_by_person(
                faces, persons, self.config.person_gate_min_size
            )
            for faces, persons in zip(faces_per_frame, persons_per_frame)
        ]

    def _detect_batch(self, batch):
        """Detect over a batch, skipping frames covered by tracking.

        Args:
            batch: List of ``(frame_number, frame)`` in order.

        Returns:
            ``(faces_per_frame, plates_per_frame)``, one list per input frame.
            Frames where detection was skipped get an empty list, which the
            tracker fills from the surrounding detections.
        """
        step = self.config.detect_every
        positions = [i for i, (num, _) in enumerate(batch) if num % step == 0]
        frames = [batch[i][1] for i in positions]

        def run(detect, enabled):
            per_frame = [[] for _ in batch]
            if not (enabled and frames):
                return per_frame
            for position, found in zip(positions, detect(frames)):
                per_frame[position] = found
            return per_frame

        return (
            run(self._detect_faces_gated, self.config.enable_face_detection),
            run(
                self.detector.detect_license_plates_batch,
                self.config.enable_license_plate_detection,
            ),
        )

    def _reset_trackers(self) -> None:
        """Drop tracks from a previous run so they cannot leak into this one."""
        if self.face_tracker:
            self.face_tracker.reset()
            self.plate_tracker.reset()

    def _anonymize_with_events(
        self,
        frame,
        frame_num: int,
        face_detections: list,
        lp_detections: list,
    ):
        """Emit detection events for already-computed detections and anonymize.

        Returns:
            The anonymized frame.
        """
        self.stats["faces_detected"] += len(face_detections)
        self.stats["license_plates_detected"] += len(lp_detections)

        if face_detections:
            self.events.emit(Event(
                event_type=EventType.FACES_DETECTED,
                timestamp=datetime.now(),
                frame_number=frame_num,
                detections={"faces": [d.to_dict() for d in face_detections]},
            ))

        if lp_detections:
            self.events.emit(Event(
                event_type=EventType.PLATES_DETECTED,
                timestamp=datetime.now(),
                frame_number=frame_num,
                detections={"license_plates": [d.to_dict() for d in lp_detections]},
            ))

        self.stats["total_detections"] += len(face_detections) + len(lp_detections)

        # Events and statistics report what the detector actually found; the
        # regions anonymized additionally cover faces it momentarily lost.
        if self.face_tracker:
            face_regions = self.face_tracker.update(face_detections, frame_num)
            lp_regions = self.plate_tracker.update(lp_detections, frame_num)
        else:
            face_regions, lp_regions = face_detections, lp_detections

        return self.anonymizer.anonymize_frame(frame, face_regions + lp_regions)

    def _detect_and_anonymize(self, frame, frame_num: int):
        """Detect on a single frame, emit events, and anonymize it.

        Used by the streaming path, where buffering frames into a batch would
        add latency to a live feed.

        Returns:
            The anonymized frame.
        """
        faces = (
            self.detector.detect_faces(frame)
            if self.config.enable_face_detection
            else []
        )
        plates = (
            self.detector.detect_license_plates(frame)
            if self.config.enable_license_plate_detection
            else []
        )

        return self._anonymize_with_events(frame, frame_num, faces, plates)

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
        self._reset_trackers()

        # Emit pipeline started event
        self.events.emit(Event(
            event_type=EventType.PIPELINE_STARTED,
            timestamp=datetime.now(),
            video_path=input_path,
        ))

        try:
            # Open input video
            with VideoReader(input_path) as reader:
                # Emit video opened event
                self.events.emit(Event(
                    event_type=EventType.VIDEO_OPENED,
                    timestamp=datetime.now(),
                    video_path=input_path,
                    total_frames=reader.frame_count,
                ))

                # Initialize output writer
                writer_class = FFmpegVideoWriter if self.use_ffmpeg else VideoWriter

                with writer_class(
                    output_path,
                    reader.fps,
                    reader.width,
                    reader.height,
                ) as writer:

                    # Detection dominates runtime, and one forward pass over
                    # several frames is measurably faster than one per frame.
                    batch_size = max(1, self.config.face_config.batch_size)

                    for batch in reader.read_frame_batch(batch_size):
                        faces_per_frame, plates_per_frame = self._detect_batch(batch)

                        for (frame_num, frame), faces, plates in zip(
                            batch, faces_per_frame, plates_per_frame
                        ):
                            # Emit frame start event
                            self.events.emit(Event(
                                event_type=EventType.FRAME_START,
                                timestamp=datetime.now(),
                                frame_number=frame_num,
                                total_frames=reader.frame_count,
                            ))

                            anonymized_frame = self._anonymize_with_events(
                                frame, frame_num, faces, plates
                            )

                            # Write frame
                            writer.write_frame(anonymized_frame)

                            # Update stats and progress
                            self.stats["frames_processed"] += 1

                            if progress_callback:
                                progress_callback(frame_num, reader.frame_count)

                            # Emit frame completed event
                            self.events.emit(Event(
                                event_type=EventType.FRAME_COMPLETED,
                                timestamp=datetime.now(),
                                frame_number=frame_num,
                                total_frames=reader.frame_count,
                                progress=(
                                    (frame_num / reader.frame_count) * 100
                                    if reader.frame_count
                                    else None
                                ),
                            ))

            logger.info(f"Video processing complete. Output: {output_path}")
            logger.info(f"Statistics: {self.stats}")

            # Emit pipeline completed event
            self.events.emit(Event(
                event_type=EventType.PIPELINE_COMPLETED,
                timestamp=datetime.now(),
                metadata=self.stats,
            ))

            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # Emit pipeline failed event
            self.events.emit(Event(
                event_type=EventType.PIPELINE_FAILED,
                timestamp=datetime.now(),
                error=str(e),
            ))
            raise

    def process_stream(
        self,
        source: str,
        output_path: str,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> Dict:
        """Anonymize a live or networked frame stream.

        A stream has no reliable length, so frame events carry no progress
        percentage. Interrupting with Ctrl-C finalizes the output rather than
        discarding what has been written.

        Args:
            source: ``webcam://0``, an RTSP/HTTP URL, or a file path.
            output_path: Path for the anonymized output video.
            max_frames: Stop after this many frames.
            max_seconds: Stop after this much wall-clock time.

        Returns:
            Dictionary with processing statistics.
        """
        logger.info(f"Starting stream anonymization: {source} -> {output_path}")
        self._reset_trackers()

        self.events.emit(Event(
            event_type=EventType.PIPELINE_STARTED,
            timestamp=datetime.now(),
            video_path=source,
        ))

        # Load models before opening the stream so --max-seconds measures
        # capture time rather than model initialization, and so a live source
        # isn't left buffering stale frames during a multi-second load.
        device = self.config.face_config.device
        if self.config.enable_face_detection:
            self.model_loader.load_face_detector(device)
        if self.config.enable_license_plate_detection:
            self.model_loader.load_license_plate_detector(device)

        try:
            with FrameStream(source) as stream:
                self.events.emit(Event(
                    event_type=EventType.STREAM_OPENED,
                    timestamp=datetime.now(),
                    video_path=source,
                    total_frames=stream.frame_count,
                    metadata={
                        "fps": stream.fps,
                        "width": stream.width,
                        "height": stream.height,
                        "is_live": stream.is_live,
                    },
                ))

                writer_class = FFmpegVideoWriter if self.use_ffmpeg else VideoWriter

                with writer_class(
                    output_path,
                    stream.fps,
                    stream.width,
                    stream.height,
                ) as writer:
                    interrupted = False

                    try:
                        for frame_num, frame in stream.read_frames(
                            max_frames=max_frames,
                            max_seconds=max_seconds,
                        ):
                            self.events.emit(Event(
                                event_type=EventType.FRAME_START,
                                timestamp=datetime.now(),
                                frame_number=frame_num,
                            ))

                            writer.write_frame(
                                self._detect_and_anonymize(frame, frame_num)
                            )
                            self.stats["frames_processed"] += 1

                            self.events.emit(Event(
                                event_type=EventType.FRAME_COMPLETED,
                                timestamp=datetime.now(),
                                frame_number=frame_num,
                                total_frames=stream.frame_count,
                            ))
                    except KeyboardInterrupt:
                        interrupted = True
                        logger.info("Interrupted, finalizing output...")

                self.events.emit(Event(
                    event_type=EventType.STREAM_CLOSED,
                    timestamp=datetime.now(),
                    video_path=source,
                ))

            logger.info(f"Stream processing complete. Output: {output_path}")
            logger.info(f"Statistics: {self.stats}")

            self.events.emit(Event(
                event_type=EventType.PIPELINE_COMPLETED,
                timestamp=datetime.now(),
                metadata={**self.stats, "interrupted": interrupted},
            ))

            return self.stats

        except Exception as e:
            logger.error(f"Stream pipeline failed: {e}")
            self.events.emit(Event(
                event_type=EventType.PIPELINE_FAILED,
                timestamp=datetime.now(),
                error=str(e),
            ))
            raise

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
