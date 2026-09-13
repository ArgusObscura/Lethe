"""Basic usage example for Lethe anonymization tool."""

from pathlib import Path
import sys

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anonymizer import AnonymizationPipeline, AnonymizationConfig


def example_basic_anonymization():
    """Example: Basic video anonymization with default settings."""
    print("Example 1: Basic Anonymization")
    print("-" * 50)

    # Create pipeline with default config
    pipeline = AnonymizationPipeline()

    # Process video (replace with actual video path)
    # stats = pipeline.process_video("input.mp4", "output.mp4")
    # print(f"Processed {stats['frames_processed']} frames")

    print("✓ Pipeline initialized (ready to process videos)")


def example_custom_config():
    """Example: Anonymization with custom configuration."""
    print("\nExample 2: Custom Configuration")
    print("-" * 50)

    # Create custom config
    config = AnonymizationConfig(
        method="blur",
        blur_kernel_size=51,  # Stronger blur
        enable_face_detection=True,
        enable_license_plate_detection=True,
    )

    # Create pipeline with custom config
    pipeline = AnonymizationPipeline(config)

    print("✓ Pipeline created with custom blur settings (kernel=51)")


def example_pixelation():
    """Example: Using pixelation instead of blur."""
    print("\nExample 3: Pixelation Anonymization")
    print("-" * 50)

    config = AnonymizationConfig(
        method="pixelate",
        pixelate_size=20,  # Larger blocks = more obvious
    )

    pipeline = AnonymizationPipeline(config)

    print("✓ Pipeline configured for pixelation (block size=20)")


def example_masking():
    """Example: Using solid color masking."""
    print("\nExample 4: Color Masking")
    print("-" * 50)

    config = AnonymizationConfig(
        method="mask",
        mask_color=(0, 0, 0),  # Black mask
    )

    pipeline = AnonymizationPipeline(config)

    print("✓ Pipeline configured for black color masking")


def example_detection_only():
    """Example: Detect objects without anonymization."""
    print("\nExample 5: Detection Only (Debug)")
    print("-" * 50)

    pipeline = AnonymizationPipeline()

    # Detect objects in a frame without anonymizing
    # detections = pipeline.detect_frame("image.jpg")
    # print(f"Found {len(detections['faces'])} faces")
    # print(f"Found {len(detections['license_plates'])} license plates")

    print("✓ Pipeline ready for detection-only analysis")


def example_batch_processing():
    """Example: Processing multiple videos."""
    print("\nExample 6: Batch Processing")
    print("-" * 50)

    config = AnonymizationConfig()
    pipeline = AnonymizationPipeline(config)

    # Example of processing multiple videos
    input_videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

    for i, video in enumerate(input_videos, 1):
        # output_path = f"output_{i}.mp4"
        # stats = pipeline.process_video(video, output_path)
        # print(f"✓ Processed {video}: {stats['frames_processed']} frames")

        print(f"Would process: {video}")


def example_with_context_manager():
    """Example: Using pipeline as context manager."""
    print("\nExample 7: Context Manager Usage")
    print("-" * 50)

    with AnonymizationPipeline() as pipeline:
        # Pipeline will auto-cleanup on exit
        # stats = pipeline.process_video("input.mp4", "output.mp4")

        print("✓ Pipeline used as context manager (auto-cleanup)")


def main():
    """Run all examples."""
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "Lethe Anonymization Tool Examples" + " " * 5 + "║")
    print("╚" + "=" * 48 + "╝")
    print()

    example_basic_anonymization()
    example_custom_config()
    example_pixelation()
    example_masking()
    example_detection_only()
    example_batch_processing()
    example_with_context_manager()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with actual videos, uncomment the lines that call:")
    print("  - pipeline.process_video(input, output)")
    print("  - pipeline.detect_frame(image)")
    print("=" * 50)


if __name__ == "__main__":
    main()
