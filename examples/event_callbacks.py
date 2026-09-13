"""Examples of using event callbacks with Lethe anonymization pipeline."""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anonymizer import AnonymizationPipeline, AnonymizationConfig
from anonymizer.events import EventType, EventLogger, EventStats


def example_basic_logging():
    """Example 1: Log all events to console."""
    print("Example 1: Basic Event Logging")
    print("=" * 50)

    config = AnonymizationConfig(method="blur")
    pipeline = AnonymizationPipeline(config)

    # Create event logger
    logger = EventLogger(verbose=True)

    # Register logger for all important events
    pipeline.on(EventType.PIPELINE_STARTED, logger)
    pipeline.on(EventType.VIDEO_OPENED, logger)
    pipeline.on(EventType.FRAME_START, logger)
    pipeline.on(EventType.FACES_DETECTED, logger)
    pipeline.on(EventType.PLATES_DETECTED, logger)
    pipeline.on(EventType.FRAME_COMPLETED, logger)
    pipeline.on(EventType.PIPELINE_COMPLETED, logger)

    print("✓ Event logger configured")
    print("✓ Process video to see events logged")
    print()


def example_progress_tracking():
    """Example 2: Track processing progress."""
    print("Example 2: Progress Tracking")
    print("=" * 50)

    config = AnonymizationConfig(method="blur")
    pipeline = AnonymizationPipeline(config)

    # Create custom progress callback
    def track_progress(event):
        if event.progress_percent is not None:
            bar_length = 40
            filled = int(bar_length * event.progress_percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r[{bar}] {event.progress_percent:.1f}%", end="", flush=True)

    # Register for frame completed events
    pipeline.on(EventType.FRAME_COMPLETED, track_progress)

    print("✓ Progress tracking configured")
    print()


def example_detection_statistics():
    """Example 3: Collect detection statistics."""
    print("Example 3: Detection Statistics")
    print("=" * 50)

    config = AnonymizationConfig(method="blur")
    pipeline = AnonymizationPipeline(config)

    # Create stats collector
    stats = EventStats()

    # Register for all relevant events
    pipeline.on(EventType.PIPELINE_STARTED, stats)
    pipeline.on(EventType.FACES_DETECTED, stats)
    pipeline.on(EventType.PLATES_DETECTED, stats)
    pipeline.on(EventType.FRAME_COMPLETED, stats)
    pipeline.on(EventType.PIPELINE_COMPLETED, stats)
    pipeline.on(EventType.PIPELINE_FAILED, stats)

    # After processing:
    # summary = stats.get_summary()
    # print(f"Total faces detected: {summary['total_faces']}")
    # print(f"Total plates detected: {summary['total_plates']}")

    print("✓ Statistics collector configured")
    print()


def example_custom_callbacks():
    """Example 4: Custom event callbacks."""
    print("Example 4: Custom Callbacks")
    print("=" * 50)

    config = AnonymizationConfig(method="blur")
    pipeline = AnonymizationPipeline(config)

    # Custom callback to alert on high face detection
    def alert_on_faces(event):
        if event.detections and len(event.detections.get("faces", [])) > 10:
            print(f"⚠️  HIGH FACE COUNT: {len(event.detections['faces'])} faces on frame {event.frame_number}")

    # Custom callback to log license plates
    def log_plates(event):
        if event.detections:
            count = len(event.detections.get("license_plates", []))
            print(f"📋 Found {count} license plate(s) on frame {event.frame_number}")

    # Custom callback for errors
    def handle_error(event):
        print(f"❌ ERROR: {event.error}")

    # Register callbacks
    pipeline.on(EventType.FACES_DETECTED, alert_on_faces)
    pipeline.on(EventType.PLATES_DETECTED, log_plates)
    pipeline.on(EventType.PIPELINE_FAILED, handle_error)

    print("✓ Custom callbacks registered")
    print()


def example_chaining():
    """Example 5: Method chaining for easy setup."""
    print("Example 5: Method Chaining")
    print("=" * 50)

    config = AnonymizationConfig(method="blur")

    # Chain event registrations
    pipeline = (AnonymizationPipeline(config)
        .on(EventType.PIPELINE_STARTED, lambda e: print("🎬 Starting..."))
        .on(EventType.VIDEO_OPENED, lambda e: print(f"📹 Opened: {e.total_frames} frames"))
        .on(EventType.PIPELINE_COMPLETED, lambda e: print("✅ Done!"))
        .on(EventType.PIPELINE_FAILED, lambda e: print(f"❌ Failed: {e.error}"))
    )

    print("✓ Pipeline configured with chained callbacks")
    print()


def example_full_workflow():
    """Example 6: Complete workflow with callbacks."""
    print("Example 6: Full Workflow")
    print("=" * 50)

    config = AnonymizationConfig(method="blur", blur_kernel_size=31)
    pipeline = AnonymizationPipeline(config)

    # Set up comprehensive event handling
    logger = EventLogger(verbose=True)
    stats = EventStats()

    # Register events
    pipeline.on(EventType.PIPELINE_STARTED, logger)
    pipeline.on(EventType.VIDEO_OPENED, logger)
    pipeline.on(EventType.FACES_DETECTED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.PLATES_DETECTED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.FRAME_COMPLETED, stats)
    pipeline.on(EventType.PIPELINE_COMPLETED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.PIPELINE_FAILED, lambda e: (logger(e), stats(e)))

    # To process a video:
    # stats = pipeline.process_video("input.mp4", "output.mp4")
    # summary = stats.get_summary()

    print("✓ Full workflow example configured")
    print("✓ Ready to process videos with comprehensive event tracking")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "Lethe Event Callbacks Examples" + " " * 10 + "║")
    print("╚" + "=" * 48 + "╝")
    print()

    example_basic_logging()
    example_progress_tracking()
    example_detection_statistics()
    example_custom_callbacks()
    example_chaining()
    example_full_workflow()

    print("=" * 50)
    print("All examples configured successfully!")
    print("\nTo use these in your code:")
    print("1. Import AnonymizationPipeline and EventType")
    print("2. Create a pipeline with config")
    print("3. Register callbacks using .on(EventType, callback)")
    print("4. Process videos as normal - events will fire automatically")
    print()
