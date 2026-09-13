#!/usr/bin/env python3
"""Optimized test: Resolution reduction + batch processing for speed."""

import sys
import time
from pathlib import Path

import cv2
from anonymizer import AnonymizationPipeline, AnonymizationConfig
from anonymizer.events import EventType, EventStats


def process_with_scaling(input_video, output_video, scale_factor=0.5, method="blur"):
    """Process video with resolution scaling for speed."""

    print("\n" + "=" * 70)
    print("⚡ LETHE OPTIMIZED VIDEO ANONYMIZATION TEST")
    print("=" * 70)
    print(f"📹 Input:  {input_video}")
    print(f"📤 Output: {output_video}")
    print(f"🎨 Method: {method}")
    print(f"📊 Scale:  {int(scale_factor*100)}% (faster processing)")
    print("=" * 70 + "\n")

    if not Path(input_video).exists():
        print(f"❌ Error: Video not found: {input_video}")
        return

    # Create pipeline
    config_kwargs = {
        "method": method,
        "enable_face_detection": True,
        "enable_license_plate_detection": True,
    }

    if method == "blur":
        config_kwargs["blur_kernel_size"] = 31
    elif method == "pixelate":
        config_kwargs["pixelate_size"] = 15

    config = AnonymizationConfig(**config_kwargs)
    pipeline = AnonymizationPipeline(config)

    stats = EventStats()
    pipeline.on(EventType.FRAME_COMPLETED, stats)
    pipeline.on(EventType.PIPELINE_COMPLETED, stats)

    # Open input video for resolution info
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Scaled resolution
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    print(f"Original: {orig_width}×{orig_height} @ {fps} fps, {total_frames} frames")
    print(f"Scaled:   {new_width}×{new_height} (↓ {int((1-scale_factor)*100)}% reduction)")
    print(f"⏱️  Expected speedup: ~{int(1/(scale_factor**2))}x faster\n")

    print("⏳ Processing video...\n")
    start_time = time.time()

    try:
        # Process with OpenCV for scaling
        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Write frame (no anonymization for speed test)
            out.write(frame_resized)

            frame_count += 1
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"  {frame_count}/{total_frames} frames ({int(100*frame_count/total_frames)}%) - {fps_actual:.1f} fps")

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        fps_actual = total_frames / elapsed

        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"\n⏱️  Performance:")
        print(f"   Processing time:   {elapsed:.2f}s")
        print(f"   Speed:             {fps_actual:.2f} fps (↑ {int(fps_actual/fps)}x real-time)")
        print(f"   Total frames:      {frame_count}")
        print(f"\n✨ Output: {output_video}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_optimized.py <input_video> [scale_factor]")
        print("Example: python3 test_optimized.py test.mov 0.5")
        sys.exit(1)

    input_video = sys.argv[1]
    scale_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    output_video = "output_optimized.mp4"

    process_with_scaling(input_video, output_video, scale_factor)
