"""Event system for Lethe anonymization pipeline.

Provides an event-driven architecture allowing users to monitor and react
to different stages of video processing.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime


class EventType(str, Enum):
    """Types of events emitted during video processing."""

    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"

    # Video processing
    VIDEO_OPENED = "video_opened"
    VIDEO_CLOSED = "video_closed"

    # Frame processing
    FRAME_START = "frame_start"
    FRAME_DETECTED = "frame_detected"
    FRAME_ANONYMIZED = "frame_anonymized"
    FRAME_WRITTEN = "frame_written"
    FRAME_COMPLETED = "frame_completed"
    FRAME_FAILED = "frame_failed"

    # Detection events
    DETECTION_STARTED = "detection_started"
    DETECTION_COMPLETED = "detection_completed"
    FACES_DETECTED = "faces_detected"
    PLATES_DETECTED = "plates_detected"

    # Anonymization events
    ANONYMIZATION_STARTED = "anonymization_started"
    ANONYMIZATION_COMPLETED = "anonymization_completed"

    # Batch processing
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_VIDEO_STARTED = "batch_video_started"
    BATCH_VIDEO_COMPLETED = "batch_video_completed"


@dataclass
class Event:
    """Represents an event in the anonymization pipeline."""

    event_type: EventType
    timestamp: datetime
    frame_number: Optional[int] = None
    total_frames: Optional[int] = None
    video_path: Optional[str] = None
    detections: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None  # 0-100
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def progress_percent(self) -> Optional[float]:
        """Get progress as percentage."""
        if self.progress is not None:
            return self.progress
        if self.frame_number is not None and self.total_frames is not None:
            return (self.frame_number / self.total_frames) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "total_frames": self.total_frames,
            "video_path": self.video_path,
            "detections": self.detections,
            "progress": self.progress_percent,
            "error": self.error,
            "metadata": self.metadata,
        }


# Type alias for callback functions
EventCallback = Callable[[Event], None]


class EventEmitter:
    """Manages event callbacks and emission."""

    def __init__(self):
        """Initialize event emitter."""
        self._callbacks: Dict[EventType, List[EventCallback]] = {}
        self._enabled = True

    def on(self, event_type: EventType, callback: EventCallback) -> 'EventEmitter':
        """Register a callback for an event type.

        Args:
            event_type: Type of event to listen for
            callback: Callable to invoke when event is emitted

        Returns:
            Self for method chaining
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
        return self

    def off(self, event_type: EventType, callback: EventCallback) -> 'EventEmitter':
        """Unregister a callback.

        Args:
            event_type: Type of event
            callback: Callback to remove

        Returns:
            Self for method chaining
        """
        if event_type in self._callbacks:
            self._callbacks[event_type] = [
                cb for cb in self._callbacks[event_type] if cb != callback
            ]
        return self

    def emit(self, event: Event) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: Event to emit
        """
        if not self._enabled:
            return

        if event.event_type in self._callbacks:
            for callback in self._callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Log callback errors but don't break processing
                    import logging
                    logging.error(f"Error in event callback: {e}")

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def clear(self) -> None:
        """Clear all registered callbacks."""
        self._callbacks.clear()

    def has_listeners(self, event_type: EventType) -> bool:
        """Check if event has registered listeners.

        Args:
            event_type: Type of event to check

        Returns:
            True if listeners exist
        """
        return event_type in self._callbacks and len(self._callbacks[event_type]) > 0


class EventLogger:
    """Simple event logger that prints events."""

    def __init__(self, verbose: bool = True):
        """Initialize logger.

        Args:
            verbose: Include detailed information
        """
        self.verbose = verbose
        self.event_count = 0

    def __call__(self, event: Event) -> None:
        """Log an event."""
        self.event_count += 1

        progress_str = ""
        if event.progress_percent is not None:
            progress_str = f" ({event.progress_percent:.1f}%)"

        message = f"[{event.event_type.value}]{progress_str}"

        if event.frame_number is not None:
            message += f" Frame {event.frame_number}"

        if event.detections and self.verbose:
            faces = len(event.detections.get("faces", []))
            plates = len(event.detections.get("license_plates", []))
            message += f" | {faces} faces, {plates} plates"

        if event.error:
            message += f" | ERROR: {event.error}"

        print(message)


class EventStats:
    """Collects statistics from events."""

    def __init__(self):
        """Initialize stats collector."""
        self.total_frames = 0
        self.processed_frames = 0
        self.total_faces = 0
        self.total_plates = 0
        self.errors = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def __call__(self, event: Event) -> None:
        """Process event and update statistics."""
        if event.event_type == EventType.PIPELINE_STARTED:
            self.start_time = event.timestamp

        elif event.event_type == EventType.PIPELINE_COMPLETED:
            self.end_time = event.timestamp

        elif event.event_type == EventType.VIDEO_OPENED:
            self.total_frames = event.total_frames or 0

        elif event.event_type == EventType.FRAME_COMPLETED:
            self.processed_frames += 1

        elif event.event_type == EventType.FACES_DETECTED:
            if event.detections:
                self.total_faces += len(event.detections.get("faces", []))

        elif event.event_type == EventType.PLATES_DETECTED:
            if event.detections:
                self.total_plates += len(event.detections.get("license_plates", []))

        elif event.event_type == EventType.PIPELINE_FAILED:
            if event.error:
                self.errors.append(event.error)

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary.

        Returns:
            Dictionary with collected statistics
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_faces": self.total_faces,
            "total_plates": self.total_plates,
            "total_detections": self.total_faces + self.total_plates,
            "errors": self.errors,
            "duration_seconds": duration,
        }
