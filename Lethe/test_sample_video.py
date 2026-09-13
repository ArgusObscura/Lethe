#!/usr/bin/env python3
"""
Test script: Run Lethe anonymization on a sample video with event monitoring.

Usage:
    python3 test_sample_video.py <input_video> [output_video] [method]

Examples:
    python3 test_sample_video.py sample.mp4
    python3 test_sample_video.py sample.mp4 output.mp4 blur
    python3 test_sample_video.py sample.mp4 output.mp4 pixelate
"""

import sys
from pathlib import Path

from anonymizer import AnonymizationPipeline, AnonymizationConfig
from anonymizer.events import EventType, EventLogger, EventStats


def main():
    """Run anonymization with event monitoring."""

    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output_anonymized.mp4"
    method = sys.argv[3] if len(sys.argv) > 3 else "blur"

    # Verify input exists
    if not Path(input_video).exists():
        print(f"❌ Error: Input video not found: {input_video}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🎬 LETHE VIDEO ANONYMIZATION TEST")
    print("=" * 70)
    print(f"📹 Input:  {input_video}")
    print(f"📤 Output: {output_video}")
    print(f"🎨 Method: {method}")
    print("=" * 70 + "\n")

    # Configure pipeline
    config_kwargs = {
        "method": method,
        "enable_face_detection": True,
        "enable_license_plate_detection": True,
    }

    if method == "blur":
        config_kwargs["blur_kernel_size"] = 31
    elif method == "pixelate":
        config_kwargs["pixelate_size"] = 15
    elif method == "mask":
        config_kwargs["mask_color"] = (0, 0, 0)

    config = AnonymizationConfig(**config_kwargs)

    # Create pipeline
    pipeline = AnonymizationPipeline(config)

    # Set up event monitoring
    logger = EventLogger(verbose=True)
    stats = EventStats()

    # Register event callbacks
    pipeline.on(EventType.PIPELINE_STARTED, lambda e: print("🚀 Pipeline started"))
    pipeline.on(EventType.VIDEO_OPENED,
                lambda e: print(f"📹 Video opened: {e.total_frames} frames @ {e.metadata.get('fps', 'unknown')} fps"))
    pipeline.on(EventType.FRAME_START, logger)
    pipeline.on(EventType.FACES_DETECTED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.PLATES_DETECTED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.FRAME_COMPLETED, lambda e: stats(e))
    pipeline.on(EventType.PIPELINE_COMPLETED, lambda e: (logger(e), stats(e)))
    pipeline.on(EventType.PIPELINE_FAILED, lambda e: print(f"❌ Pipeline failed: {e.error}"))

    # Process video
    try:
        print("\n⏳ Processing video...\n")
        stats_dict = pipeline.process_video(input_video, output_video)

        # Print summary
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)

        summary = stats.get_summary()
        print(f"\n📊 Statistics:")
        print(f"   Total frames:      {summary['total_frames']}")
        print(f"   Processed frames:  {summary['processed_frames']}")
        print(f"   Faces detected:    {summary['total_faces']}")
        print(f"   Plates detected:   {summary['total_plates']}")
        print(f"   Total detections:  {summary['total_detections']}")

        if summary['duration_seconds']:
            fps = summary['processed_frames'] / summary['duration_seconds']
            print(f"   Processing time:   {summary['duration_seconds']:.2f}s")
            print(f"   Speed:             {fps:.2f} fps")

        if summary['errors']:
            print(f"\n⚠️  Errors occurred:")
            for error in summary['errors']:
                print(f"   - {error}")

        print(f"\n✨ Anonymized video saved to: {output_video}")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
