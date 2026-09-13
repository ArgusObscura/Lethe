"""Real-time progress display and event logging for CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from tqdm import tqdm
from loguru import logger

from anonymizer.events import Event, EventType


class ProgressDisplay:
    """Real-time progress bar and statistics display."""

    def __init__(self, total_frames: int, output_file: Optional[str] = None, quiet: bool = False):
        """Initialize progress display.

        Args:
            total_frames: Total number of frames to process
            output_file: Optional file to save event log
            quiet: Suppress progress output
        """
        self.total_frames = total_frames
        self.quiet = quiet
        self.output_file = output_file

        # Progress tracking
        self.frames_processed = 0
        self.start_time = None

        # Statistics
        self.total_faces = 0
        self.total_plates = 0
        self.total_detections = 0

        # Progress bar
        self.pbar = None

        # Logging
        self.events = []
        self.log_file = None

        if output_file:
            self.log_file = Path(output_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_pipeline_started(self, event: Event) -> None:
        """Handle pipeline started event."""
        self.start_time = event.timestamp
        self._log_event(event)

        if not self.quiet:
            print("\n🚀 Pipeline started")
            print(f"📊 Processing {self.total_frames} frames...")

    def on_video_opened(self, event: Event) -> None:
        """Handle video opened event."""
        self._log_event(event)

        if not self.quiet and event.total_frames:
            print(f"📹 Video: {event.total_frames} frames @ {event.metadata.get('fps', 'unknown')} fps")

            # Create progress bar
            self.pbar = tqdm(
                total=self.total_frames,
                desc="Processing",
                unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

    def on_frame_completed(self, event: Event) -> None:
        """Handle frame completed event."""
        self.frames_processed += 1
        self._log_event(event)

        if self.pbar and not self.quiet:
            self.pbar.update(1)

            # Update description with stats
            if self.frames_processed % 10 == 0:
                self.pbar.set_description(
                    f"Processing | Faces: {self.total_faces} | Plates: {self.total_plates}"
                )

    def on_faces_detected(self, event: Event) -> None:
        """Handle faces detected event."""
        if event.detections:
            count = len(event.detections.get("faces", []))
            self.total_faces += count
            self.total_detections += count
        self._log_event(event)

    def on_plates_detected(self, event: Event) -> None:
        """Handle plates detected event."""
        if event.detections:
            count = len(event.detections.get("license_plates", []))
            self.total_plates += count
            self.total_detections += count
        self._log_event(event)

    def on_pipeline_completed(self, event: Event) -> None:
        """Handle pipeline completed event."""
        self._log_event(event)

        if self.pbar and not self.quiet:
            self.pbar.close()

        # Save log file
        if self.log_file:
            self._save_log(event)

    def on_pipeline_failed(self, event: Event) -> None:
        """Handle pipeline failed event."""
        self._log_event(event)

        if self.pbar and not self.quiet:
            self.pbar.close()

        if not self.quiet:
            print(f"\n❌ Pipeline failed: {event.error}")

        # Save log file even on failure
        if self.log_file:
            self._save_log(event)

    def _log_event(self, event: Event) -> None:
        """Log event to memory."""
        self.events.append({
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "frame": event.frame_number,
            "progress": event.progress_percent,
            "detections": event.detections,
            "error": event.error,
        })

    def _save_log(self, final_event: Event) -> None:
        """Save event log to file."""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "total_frames": self.total_frames,
                "frames_processed": self.frames_processed,
                "events": self.events,
                "summary": {
                    "faces": self.total_faces,
                    "plates": self.total_plates,
                    "total_detections": self.total_detections,
                    "duration": (final_event.timestamp - self.start_time).total_seconds()
                    if self.start_time else None,
                },
            }

            with open(self.log_file, "w") as f:
                json.dump(log_data, f, indent=2)

            logger.info(f"Event log saved to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save event log: {e}")

    def print_summary(self) -> None:
        """Print final processing summary."""
        if self.quiet:
            return

        duration = (self.events[-1].get("timestamp") if self.events else None)

        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Frames processed:  {self.frames_processed}/{self.total_frames}")
        print(f"   Faces detected:    {self.total_faces}")
        print(f"   Plates detected:   {self.total_plates}")
        print(f"   Total detections:  {self.total_detections}")

        if self.log_file:
            print(f"\n📝 Event log: {self.log_file}")

        print("=" * 70 + "\n")
