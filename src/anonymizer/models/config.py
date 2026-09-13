"""Configuration models for Lethe anonymization tool."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
