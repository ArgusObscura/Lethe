"""Command-line interface for Lethe anonymization tool."""

import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from loguru import logger

from .core import AnonymizationPipeline
from .models.config import AnonymizationConfig, AnonymizationMethod


# Configure logging
logger.remove()
logger.add(
    lambda msg: click.echo(msg, err=False),
    format="<level>{level: <8}</level> | {message}",
    level="INFO"
)


class ConfigParamType(click.ParamType):
    """Custom Click parameter type for configuration files."""

    name = "config"

    def convert(self, value, param, ctx):
        """Convert and validate config file."""
        if value is None:
            return None

        config_path = Path(value)
        if not config_path.exists():
            self.fail(f"Config file not found: {value}", param, ctx)

        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.fail(f"Failed to load config file: {e}", param, ctx)


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="Lethe")
def cli():
    """Lethe - Vehicle Camera Video Anonymization Tool

    Anonymize sensitive objects (faces, license plates) in vehicle camera videos
    for autonomous driving R&D and privacy protection.

    Examples:

        lethe process input.mp4 -o output.mp4

        lethe process input.mp4 -o output.mp4 --method pixelate

        lethe batch input_directory/ -o output_directory/
    """
    if len(sys.argv) == 1:
        click.echo(cli.get_help(click.Context(cli)))


@cli.command()
@click.argument("input_video", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    required=True,
    help="Output video path"
)
@click.option(
    "-m", "--method",
    type=click.Choice(["blur", "pixelate", "mask"], case_sensitive=False),
    default="blur",
    help="Anonymization method [default: blur]"
)
@click.option(
    "--blur-kernel",
    type=int,
    default=31,
    help="Kernel size for blur (odd number) [default: 31]"
)
@click.option(
    "--pixelate-size",
    type=int,
    default=15,
    help="Block size for pixelation [default: 15]"
)
@click.option(
    "--mask-color",
    type=str,
    default="0,0,0",
    help="RGB color for masking [default: 0,0,0 (black)]"
)
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Detection confidence threshold 0-1 [default: 0.5]"
)
@click.option(
    "--faces/--no-faces",
    default=True,
    help="Enable face detection [default: enabled]"
)
@click.option(
    "--plates/--no-plates",
    default=True,
    help="Enable license plate detection [default: enabled]"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Processing device [default: cpu]"
)
@click.option(
    "-c", "--config",
    type=ConfigParamType(),
    help="Configuration file (YAML)"
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress progress output"
)
def process(
    input_video: str,
    output: str,
    method: str,
    blur_kernel: int,
    pixelate_size: int,
    mask_color: str,
    confidence: float,
    faces: bool,
    plates: bool,
    device: str,
    config: Optional[dict],
    quiet: bool,
):
    """Process a single video and anonymize sensitive objects.

    Examples:

        lethe process input.mp4 -o output.mp4

        lethe process video.mp4 -o anon.mp4 --method pixelate --pixelate-size 20

        lethe process video.mp4 -o anon.mp4 --config config.yaml
    """
    try:
        # Parse mask color
        try:
            mask_rgb = tuple(map(int, mask_color.split(",")))
            if len(mask_rgb) != 3:
                raise ValueError("Color must be 3 RGB values")
        except ValueError as e:
            raise click.BadParameter(f"Invalid color format: {e}")

        # Create configuration
        if config:
            # Load from YAML config
            click.echo("📋 Loading configuration from file...", err=True)
            anonymization_config = AnonymizationConfig(**config.get("anonymization", {}))
        else:
            # Build from command-line arguments
            from .models.config import DetectionConfig

            anonymization_config = AnonymizationConfig(
                method=method,
                blur_kernel_size=blur_kernel,
                pixelate_size=pixelate_size,
                mask_color=mask_rgb,
                enable_face_detection=faces,
                enable_license_plate_detection=plates,
                face_config=DetectionConfig(
                    confidence_threshold=confidence,
                    device=device,
                ),
                license_plate_config=DetectionConfig(
                    confidence_threshold=confidence,
                    device=device,
                ),
            )

        # Show configuration
        click.echo(f"🎬 Input: {input_video}", err=True)
        click.echo(f"📹 Output: {output}", err=True)
        click.echo(f"🎨 Method: {anonymization_config.method}", err=True)
        click.echo(f"👤 Face detection: {'✓' if anonymization_config.enable_face_detection else '✗'}", err=True)
        click.echo(f"📋 Plate detection: {'✓' if anonymization_config.enable_license_plate_detection else '✗'}", err=True)
        click.echo()

        # Initialize pipeline
        pipeline = AnonymizationPipeline(anonymization_config)

        # Process video
        with click.progressbar(length=100, label="Processing", show_pos=True, show_eta=True) as bar:
            def progress_callback(current, total):
                if not quiet:
                    bar.update(int((current / total) * 100) - bar.pos)

            stats = pipeline.process_video(input_video, output, progress_callback)

        # Show results
        click.echo()
        click.secho("✅ Processing Complete!", fg="green", bold=True)
        click.echo()
        click.echo(f"📊 Statistics:")
        click.echo(f"  • Frames processed: {stats['frames_processed']}")
        click.echo(f"  • Faces detected: {stats['faces_detected']}")
        click.echo(f"  • License plates detected: {stats['license_plates_detected']}")
        click.echo(f"  • Total objects anonymized: {stats['total_detections']}")
        click.echo()

        pipeline.cleanup()

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        logger.exception("Processing failed")
        sys.exit(1)


@cli.command()
@click.argument("input_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "-o", "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory"
)
@click.option(
    "-m", "--method",
    type=click.Choice(["blur", "pixelate", "mask"], case_sensitive=False),
    default="blur",
    help="Anonymization method [default: blur]"
)
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Detection confidence threshold [default: 0.5]"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Processing device [default: cpu]"
)
@click.option(
    "-c", "--config",
    type=ConfigParamType(),
    help="Configuration file (YAML)"
)
@click.option(
    "-p", "--pattern",
    default="*.mp4",
    help="File pattern to match [default: *.mp4]"
)
def batch(
    input_directory: str,
    output: str,
    method: str,
    confidence: float,
    device: str,
    config: Optional[dict],
    pattern: str,
):
    """Batch process multiple videos in a directory.

    Examples:

        lethe batch input/ -o output/

        lethe batch videos/ -o anon/ --pattern "*.mp4" --method pixelate
    """
    try:
        input_dir = Path(input_directory)
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find video files
        video_files = list(input_dir.glob(pattern))

        if not video_files:
            click.secho(f"⚠️  No videos found matching '{pattern}'", fg="yellow")
            return

        click.echo(f"📁 Found {len(video_files)} video(s) to process", err=True)
        click.echo()

        # Create configuration
        from .models.config import DetectionConfig

        if config:
            anonymization_config = AnonymizationConfig(**config.get("anonymization", {}))
        else:
            anonymization_config = AnonymizationConfig(
                method=method,
                face_config=DetectionConfig(
                    confidence_threshold=confidence,
                    device=device,
                ),
                license_plate_config=DetectionConfig(
                    confidence_threshold=confidence,
                    device=device,
                ),
            )

        # Initialize pipeline
        pipeline = AnonymizationPipeline(anonymization_config)

        # Process each video
        total_stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "license_plates_detected": 0,
            "total_detections": 0,
        }

        with click.progressbar(video_files, label="Batch processing", show_pos=True) as bar:
            for video_file in bar:
                output_file = output_dir / video_file.name

                try:
                    stats = pipeline.process_video(str(video_file), str(output_file))

                    # Accumulate statistics
                    for key in total_stats:
                        total_stats[key] += stats.get(key, 0)

                    click.echo(f"  ✓ {video_file.name}")

                except Exception as e:
                    click.echo(f"  ✗ {video_file.name}: {e}", err=True)

        # Show summary
        click.echo()
        click.secho("✅ Batch processing complete!", fg="green", bold=True)
        click.echo()
        click.echo(f"📊 Total Statistics:")
        click.echo(f"  • Videos processed: {len(video_files)}")
        click.echo(f"  • Total frames: {total_stats['frames_processed']}")
        click.echo(f"  • Faces detected: {total_stats['faces_detected']}")
        click.echo(f"  • License plates detected: {total_stats['license_plates_detected']}")
        click.echo(f"  • Total objects anonymized: {total_stats['total_detections']}")
        click.echo()

        pipeline.cleanup()

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        logger.exception("Batch processing failed")
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Detection confidence threshold [default: 0.5]"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Processing device [default: cpu]"
)
def detect(image_path: str, confidence: float, device: str):
    """Detect objects in an image without anonymization.

    Useful for testing detection accuracy and visualizing detected regions.

    Examples:

        lethe detect image.jpg

        lethe detect photo.png --confidence 0.7
    """
    try:
        from .models.config import DetectionConfig

        config = AnonymizationConfig(
            face_config=DetectionConfig(
                confidence_threshold=confidence,
                device=device,
            ),
            license_plate_config=DetectionConfig(
                confidence_threshold=confidence,
                device=device,
            ),
        )

        pipeline = AnonymizationPipeline(config)

        click.echo(f"🔍 Detecting objects in: {image_path}", err=True)
        detections = pipeline.detect_frame(image_path)

        click.echo()
        click.secho("✅ Detection Results:", fg="green", bold=True)
        click.echo()

        faces = detections.get("faces", [])
        click.echo(f"👤 Faces detected: {len(faces)}")
        for i, face in enumerate(faces, 1):
            click.echo(f"   {i}. Confidence: {face['confidence']:.2%}")

        plates = detections.get("license_plates", [])
        click.echo(f"📋 License plates detected: {len(plates)}")
        for i, plate in enumerate(plates, 1):
            click.echo(f"   {i}. Confidence: {plate['confidence']:.2%}")

        click.echo()
        click.echo(f"Total objects: {len(faces) + len(plates)}")

        pipeline.cleanup()

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        logger.exception("Detection failed")
        sys.exit(1)


@cli.command()
@click.option(
    "-m", "--method",
    type=click.Choice(["blur", "pixelate", "mask"], case_sensitive=False),
    default="blur",
    help="Anonymization method to show"
)
def show_config(method: str):
    """Show configuration template for a given method.

    Helps you create configuration files with all available options.

    Examples:

        lethe show-config

        lethe show-config --method pixelate
    """
    config = AnonymizationConfig(method=method)

    config_dict = {
        "anonymization": {
            "method": config.method,
            "blur_kernel_size": config.blur_kernel_size,
            "pixelate_size": config.pixelate_size,
            "mask_color": config.mask_color,
            "enable_face_detection": config.enable_face_detection,
            "enable_license_plate_detection": config.enable_license_plate_detection,
        },
        "detection": {
            "face": {
                "confidence_threshold": config.face_config.confidence_threshold,
                "iou_threshold": config.face_config.iou_threshold,
                "device": config.face_config.device,
                "batch_size": config.face_config.batch_size,
            },
            "license_plate": {
                "confidence_threshold": config.license_plate_config.confidence_threshold,
                "iou_threshold": config.license_plate_config.iou_threshold,
                "device": config.license_plate_config.device,
                "batch_size": config.license_plate_config.batch_size,
            },
        },
    }

    click.echo(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.secho(f"\n❌ Unexpected error: {e}", fg="red", err=True)
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
