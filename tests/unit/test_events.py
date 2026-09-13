"""Unit tests for event system."""

import pytest
from datetime import datetime

from anonymizer.events import (
    Event, EventType, EventEmitter, EventLogger, EventStats
)


class TestEvent:
    """Test Event class."""

    def test_event_creation(self):
        """Test creating an event."""
        now = datetime.now()
        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=now,
            frame_number=42,
            total_frames=100,
        )

        assert event.event_type == EventType.FRAME_COMPLETED
        assert event.timestamp == now
        assert event.frame_number == 42
        assert event.total_frames == 100

    def test_event_progress_calculation(self):
        """Test progress percentage calculation."""
        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
            frame_number=50,
            total_frames=100,
        )

        assert event.progress_percent == 50.0

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_type=EventType.PIPELINE_STARTED,
            timestamp=datetime.now(),
            video_path="test.mp4",
        )

        event_dict = event.to_dict()
        assert event_dict["event_type"] == "pipeline_started"
        assert event_dict["video_path"] == "test.mp4"
        assert "timestamp" in event_dict


class TestEventEmitter:
    """Test EventEmitter class."""

    def test_register_callback(self):
        """Test registering a callback."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        emitter.on(EventType.FRAME_COMPLETED, callback)

        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called) == 1
        assert called[0] == event

    def test_multiple_callbacks(self):
        """Test multiple callbacks for same event."""
        emitter = EventEmitter()
        called1 = []
        called2 = []

        def callback1(event):
            called1.append(event)

        def callback2(event):
            called2.append(event)

        emitter.on(EventType.FACES_DETECTED, callback1)
        emitter.on(EventType.FACES_DETECTED, callback2)

        event = Event(
            event_type=EventType.FACES_DETECTED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called1) == 1
        assert len(called2) == 1

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        emitter.on(EventType.FRAME_COMPLETED, callback)
        emitter.off(EventType.FRAME_COMPLETED, callback)

        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called) == 0

    def test_method_chaining(self):
        """Test method chaining."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        result = emitter.on(EventType.FRAME_COMPLETED, callback)

        assert result is emitter
        assert len(called) == 0

    def test_disable_events(self):
        """Test disabling events."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        emitter.on(EventType.FRAME_COMPLETED, callback)
        emitter.disable()

        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called) == 0

    def test_enable_events(self):
        """Test enabling events after disabling."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        emitter.on(EventType.FRAME_COMPLETED, callback)
        emitter.disable()
        emitter.enable()

        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called) == 1

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        emitter = EventEmitter()
        called = []

        def callback(event):
            called.append(event)

        emitter.on(EventType.FRAME_COMPLETED, callback)
        emitter.clear()

        event = Event(
            event_type=EventType.FRAME_COMPLETED,
            timestamp=datetime.now(),
        )
        emitter.emit(event)

        assert len(called) == 0

    def test_has_listeners(self):
        """Test checking for listeners."""
        emitter = EventEmitter()

        def callback(event):
            pass

        assert not emitter.has_listeners(EventType.FRAME_COMPLETED)

        emitter.on(EventType.FRAME_COMPLETED, callback)
        assert emitter.has_listeners(EventType.FRAME_COMPLETED)


class TestEventStats:
    """Test EventStats class."""

    def test_collect_statistics(self):
        """Test collecting statistics from events."""
        stats = EventStats()

        # Simulate events
        stats(Event(
            event_type=EventType.PIPELINE_STARTED,
            timestamp=datetime.now(),
        ))

        stats(Event(
            event_type=EventType.VIDEO_OPENED,
            timestamp=datetime.now(),
            total_frames=100,
        ))

        stats(Event(
            event_type=EventType.FACES_DETECTED,
            timestamp=datetime.now(),
            detections={"faces": [{}, {}]},
        ))

        stats(Event(
            event_type=EventType.PLATES_DETECTED,
            timestamp=datetime.now(),
            detections={"license_plates": [{}]},
        ))

        summary = stats.get_summary()

        assert summary["total_frames"] == 100
        assert summary["total_faces"] == 2
        assert summary["total_plates"] == 1
        assert summary["total_detections"] == 3

    def test_error_collection(self):
        """Test collecting errors."""
        stats = EventStats()

        stats(Event(
            event_type=EventType.PIPELINE_FAILED,
            timestamp=datetime.now(),
            error="Test error",
        ))

        summary = stats.get_summary()

        assert len(summary["errors"]) == 1
        assert summary["errors"][0] == "Test error"

    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = EventStats()

        start = datetime.now()
        stats(Event(
            event_type=EventType.PIPELINE_STARTED,
            timestamp=start,
        ))

        # Simulate some processing
        import time
        time.sleep(0.01)  # 10ms

        end = datetime.now()
        stats(Event(
            event_type=EventType.PIPELINE_COMPLETED,
            timestamp=end,
        ))

        summary = stats.get_summary()
        assert summary["duration_seconds"] > 0
