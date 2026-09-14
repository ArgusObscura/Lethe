"""Configuration models for Lethe anonymization tool."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class AnonymizationMethod(str, Enum):
    """Available anonymization techniques."""
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"


class ObjectType(str, Enum):
    """Types of objects to detect and anonymize."""
    FACE = "face"
    LICENSE_PLATE = "license_plate"


class DetectionConfig(BaseModel):
    """Configuration for object detection."""

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections (0-1)"
    )
    iou_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS (0-1)"
    )
    device: str = Field(
        default="cpu",
        description="Device to run detection on: 'cpu' or 'cuda'"
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Batch size for detection processing"
    )
    inference_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Longest edge the detector sees, in pixels (YOLO's default is 640). "
            "Frames are scaled to this before detection, so on high-resolution "
            "footage a low value shrinks distant faces below what the model can "
            "resolve. Raising it finds more faces and costs proportionally more "
            "time."
        )
    )


class AnonymizationConfig(BaseModel):
    """Configuration for anonymization process."""

    method: AnonymizationMethod = Field(
        default=AnonymizationMethod.BLUR,
        description="Anonymization method to use"
    )
    blur_kernel_size: int = Field(
        default=31,
        ge=3,
        description="Kernel size for Gaussian blur (must be odd)"
    )
    pixelate_size: int = Field(
        default=15,
        gt=0,
        description="Pixel block size for pixelation"
    )
    mask_color: tuple = Field(
        default=(0, 0, 0),
        description="RGB color for masking (black by default)"
    )
    track_detections: bool = Field(
        default=True,
        description=(
            "Bridge frames where the detector loses a face it had found. "
            "Without it every dropout leaks an unblurred frame."
        )
    )
    detect_every: int = Field(
        default=1,
        ge=1,
        description=(
            "Run detection only every Nth frame and let tracking cover the rest. "
            "Inference dominates runtime, so this divides it directly, at the "
            "cost of recall on anything moving fast. Requires track_detections."
        )
    )
    track_persistence: int = Field(
        default=10,
        ge=0,
        description="Frames to keep anonymizing a region after its last detection"
    )
    box_padding: float = Field(
        default=0.15,
        ge=0.0,
        description=(
            "Fraction to expand each detection box before anonymizing. "
            "Detectors return tight boxes that leave hair and jaw edges exposed."
        )
    )

    # Detection-specific configs
    face_config: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Configuration for face detection"
    )
    license_plate_config: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Configuration for license plate detection"
    )

    # Processing options
    enable_face_detection: bool = Field(
        default=True,
        description="Enable face detection and anonymization"
    )
    enable_license_plate_detection: bool = Field(
        default=True,
        description="Enable license plate detection and anonymization"
    )
    smooth_boxes: bool = Field(
        default=True,
        description="Apply temporal smoothing to detection boxes"
    )

    @model_validator(mode="after")
    def _skipping_requires_tracking(self) -> "AnonymizationConfig":
        """Skipping frames without tracking would anonymize only every Nth one."""
        if self.detect_every > 1 and not self.track_detections:
            raise ValueError(
                "detect_every > 1 needs track_detections enabled, otherwise "
                "frames between detections are left unanonymized"
            )
        return self

    class Config:
        use_enum_values = True


class VideoConfig(BaseModel):
    """Configuration for video I/O."""

    output_format: str = Field(
        default="mp4",
        description="Output video format: mp4, avi, mov, mkv"
    )
    codec: Optional[str] = Field(
        default=None,
        description="Video codec to use (auto-detect if None)"
    )
    crf: int = Field(
        default=23,
        ge=0,
        le=51,
        description="Quality level for H.264 (0=best, 51=worst)"
    )
    preserve_original: bool = Field(
        default=True,
        description="Keep original frame rate and resolution"
    )

    class Config:
        use_enum_values = True
