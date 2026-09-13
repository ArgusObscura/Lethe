# Lethe

**Lethe**: A comprehensive tool for anonymizing sensitive objects (faces, license plates) in vehicle camera videos for autonomous driving R&D.

Named after the Greek mythological river of forgetfulness, Lethe erases identifiable information from your research footage while preserving video quality.

---

## Project Overview

### Purpose

Lethe addresses a critical need in autonomous driving research: **protecting privacy while leveraging real-world vehicle camera data**. When training perception systems and developing autonomous driving algorithms, researchers must process extensive video footage from vehicle cameras. This data often contains sensitive information:

- **Human faces** of drivers and pedestrians
- **License plates** of vehicles
- **Personal vehicle identifiers**

Lethe automatically detects and anonymizes these elements before videos are shared for research, published, or analyzed collaboratively.

### Key Use Cases

-  **Autonomous Driving R&D** - Anonymize perception training data
-  **Dataset Curation** - Prepare public datasets from private recordings
-  **Computer Vision Research** - Create privacy-compliant benchmark datasets
-  **Data Protection** - Comply with GDPR/privacy regulations
-  **Collaborative Research** - Share data safely across organizations

### Why Lethe?

The name comes from Greek mythology: **Lethe** is the river of forgetfulness. Those who drank from it forgot their past. Similarly, Lethe helps you "forget" (anonymize) sensitive information in your videos.

---

##  Features

### Face Detection & Anonymization
- YOLOv8-face model for human face detection
- High accuracy across lighting conditions and angles
- Confidence-threshold configurable (default: 0.5)

### License Plate Detection & Anonymization
- YOLOv8 trained on license plate recognition
- Supports multiple plate formats and regions
- Fine-tuning support for custom plate types

### Multiple Anonymization Techniques
1. **Blur** (Default) - Gaussian blur for natural appearance
2. **Pixelate** - Block pixelation for stronger privacy
3. **Mask** - Solid color replacement for complete obscuring

### Flexible Video Support
- **Formats**: MP4 (H.264), H.265 (HEVC), AVI, MOV, MKV
- **Resolutions**: 720p, 1080p, 4K, and any custom size
- **Frame Sequences**: Process individual PNG/JPG frames
- **Codec Preservation**: Maintain original codec and quality

### Multiple Deployment Options
- **CLI Tool** - Simple command-line interface
- **Python Library** - Import and use in your code
- **REST API** - HTTP service for distributed processing
- **Batch Processing** - Process thousands of videos efficiently

### Highly Configurable
- Per-object detection settings
- Customizable anonymization intensity
- Flexible confidence thresholds
- YAML/JSON configuration support

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ArgusObscura/Lethe.git
cd Lethe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Lethe (development mode)
pip install -e .
```

### Basic Usage

#### Python Library (Phase 1 - Ready Now)

```python
from anonymizer import AnonymizationPipeline, AnonymizationConfig

# Create configuration
config = AnonymizationConfig(
    method="blur",           # blur, pixelate, or mask
    blur_kernel_size=31,     # Higher = stronger blur
    enable_face_detection=True,
    enable_license_plate_detection=True,
)

# Initialize pipeline
pipeline = AnonymizationPipeline(config)

# Process video
stats = pipeline.process_video("input_video.mp4", "output_video.mp4")

print(f"Processed {stats['frames_processed']} frames")
print(f"Detected {stats['faces_detected']} faces")
print(f"Detected {stats['license_plates_detected']} license plates")
```

#### Command Line (Phase 2 - In Development)

```bash
# Basic anonymization
lethe process input.mp4 -o output.mp4

# With custom settings
lethe process input.mp4 \
  -o output.mp4 \
  --method pixelate \
  --pixelate-size 20 \
  --confidence 0.6

# Batch processing
lethe batch input_directory/ -o output_directory/

# Configuration file
lethe process input.mp4 -o output.mp4 --config config.yaml
```

#### REST API (Phase 4 - Planned)

```bash
# Start server
python -m anonymizer.api.server

# In another terminal
curl -X POST http://localhost:8000/anonymize \
  -F "video=@input.mp4" \
  -F "method=blur"
```

---

## Anonymization Methods

### Blur (Default)
**Best for**: Published papers, general R&D, professional appearance

```python
config = AnonymizationConfig(
    method="blur",
    blur_kernel_size=31,  # Higher = stronger blur
)
```

- Preserves context while hiding identity
- Natural appearance
- Reversibility: Low (harder to reverse)

### Pixelate
**Best for**: Strong privacy requirements, obvious anonymization

```python
config = AnonymizationConfig(
    method="pixelate",
    pixelate_size=15,  # Larger = larger blocks
)
```

- Block-based pixelation
- More obvious anonymization
- Reversibility: Very Low (even harder to reverse)

### Mask
**Best for**: Maximum privacy, sensitive footage

```python
config = AnonymizationConfig(
    method="mask",
    mask_color=(0, 0, 0),  # Black
)
```

- Solid color replacement
- Removes all visual information
- Reversibility: None (irreversible)

---

## Configuration

Create a `config.yaml` file for advanced control:

```yaml
anonymization:
  method: blur              # blur | pixelate | mask
  blur_kernel_size: 31      # Odd numbers only
  pixelate_size: 15
  mask_color: [0, 0, 0]     # BGR color
  enable_face_detection: true
  enable_license_plate_detection: true
  smooth_boxes: true        # Temporal smoothing

detection:
  face:
    confidence_threshold: 0.5
    iou_threshold: 0.4
    device: cpu             # cpu | cuda
    batch_size: 8
  
  license_plate:
    confidence_threshold: 0.5
    iou_threshold: 0.4
    device: cpu
    batch_size: 8

video:
  output_format: mp4        # mp4 | avi | mov | mkv
  crf: 23                   # 0 (best) to 51 (worst)
  preserve_original: true   # Keep original resolution/fps
```

Use it:

```python
from anonymizer import AnonymizationPipeline
import yaml

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

pipeline = AnonymizationPipeline(AnonymizationConfig(**config_dict))
```

---

## Performance

### Detection Speed
| Method | GPU | CPU |
|--------|-----|-----|
| Face Detection | 15-30 fps | 2-5 fps |
| License Plate | 10-20 fps | 1-3 fps |
| Both (Combined) | 8-15 fps | 0.5-1 fps |

### Memory Usage
- **Model Memory**: ~500MB (both models loaded)
- **Runtime Memory**: 2-4GB (depends on resolution)

### Recommendations
- Use GPU for production (10-100x faster)
- Process 1080p for optimal accuracy
- Use batch processing for multiple videos

---

## Project Structure

```
Lethe/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── pytest.ini                 # Test configuration
├── PHASE1_SUMMARY.md         # Phase 1 deliverables
│
├── src/anonymizer/           # Main library
│   ├── __init__.py           # Public API
│   ├── core.py               # Main pipeline
│   ├── detector.py           # Object detection
│   ├── anonymizer.py         # Anonymization methods
│   │
│   ├── models/               # Model management
│   │   ├── config.py         # Pydantic configurations
│   │   └── loader.py         # Model loading/caching
│   │
│   ├── video/                # Video I/O
│   │   ├── reader.py         # Frame extraction
│   │   ├── writer.py         # Video output
│   │   └── utils.py          # Helper functions
│   │
│   ├── cli.py                # CLI interface (Phase 2)
│   └── api/                  # REST API (Phase 4)
│       ├── server.py
│       └── routes.py
│
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── unit/                # Unit tests
│   │   ├── test_anonymizer.py
│   │   └── test_video.py
│   └── integration/         # Integration tests (Phase 2+)
│
├── examples/                # Usage examples
│   └── basic_usage.py
│
└── docs/                    # Documentation (Phase 5)
```

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src/anonymizer --cov-report=html
```

### Run Specific Tests
```bash
pytest tests/unit/test_anonymizer.py -v
pytest tests/unit/test_video.py::TestVideoReader -v
```

**Current Status**: 24 unit tests, all passing 

---

## Development Roadmap

### Phase 1: Core Detection & Anonymization (COMPLETE)
- [x] YOLOv8 face detection
- [x] YOLOv8 license plate detection
- [x] Blur anonymization
- [x] Pixelate anonymization
- [x] Mask anonymization
- [x] Video reader (FFmpeg + OpenCV)
- [x] Video writer (codec preservation)
- [x] Configuration system
- [x] Unit tests (24 tests)
- [x] Documentation

### Phase 2: CLI Tool (IN PROGRESS)
- [ ] Click/Typer CLI interface
- [ ] Batch video processing
- [ ] Progress bars (tqdm)
- [ ] Config file support
- [ ] Logging and debugging
- [ ] Integration tests

### Phase 3: Python Library Refinement
- [ ] Public API finalization
- [ ] Event callbacks/hooks
- [ ] Streaming support
- [ ] Example notebooks
- [ ] Advanced configuration guide

### Phase 4: REST API Service
- [ ] FastAPI server
- [ ] File upload/download
- [ ] Job queue and tracking
- [ ] Async processing
- [ ] Docker containerization
- [ ] API documentation

### Phase 5: Production Ready
- [ ] Comprehensive test suite (80%+ coverage)
- [ ] Performance optimization
- [ ] GitHub Actions CI/CD
- [ ] Documentation site (MkDocs)
- [ ] Release process
- [ ] Contributing guidelines
- [ ] Community setup

---

## System Requirements

- **Python**: 3.9 or higher
- **OS**: Linux, macOS, Windows
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: NVIDIA CUDA (optional, 10-100x faster)
- **FFmpeg**: Required for optimal codec support

### Optional Dependencies

```bash
# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For advanced video formats
brew install ffmpeg  # macOS
apt install ffmpeg   # Ubuntu/Debian
choco install ffmpeg # Windows
```

---

## Usage Examples

### Example 1: Simple Face Anonymization
```python
from anonymizer import AnonymizationPipeline

pipeline = AnonymizationPipeline()
pipeline.process_video("dashcam.mp4", "dashcam_anonymized.mp4")
```

### Example 2: Strong Privacy Pixelation
```python
from anonymizer import AnonymizationPipeline, AnonymizationConfig

config = AnonymizationConfig(
    method="pixelate",
    pixelate_size=25,  # Larger blocks = stronger privacy
)
pipeline = AnonymizationPipeline(config)
pipeline.process_video("video.mp4", "video_pixelated.mp4")
```

### Example 3: Process Only Faces
```python
config = AnonymizationConfig(
    enable_face_detection=True,
    enable_license_plate_detection=False,  # Skip plates
)
pipeline = AnonymizationPipeline(config)
```

### Example 4: Detection Only (No Anonymization)
```python
pipeline = AnonymizationPipeline()
detections = pipeline.detect_frame("image.jpg")
print(f"Found {len(detections['faces'])} faces")
print(f"Found {len(detections['license_plates'])} plates")
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we need help with:**
- [ ] Custom license plate detection models
- [ ] Additional anonymization techniques
- [ ] Performance optimization
- [ ] Documentation improvements
- [ ] Test coverage expansion
- [ ] Real-world dataset contributions

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### Citation

If you use Lethe in your research, please cite:

```bibtex
@software{lethe2024,
  title={Lethe: Vehicle Camera Video Anonymization Tool},
  author={ArgusObscura},
  url={https://github.com/ArgusObscura/Lethe},
  year={2024},
  note={Anonymous video processing for autonomous driving R&D}
}
```

---

## Support & Community

- **Documentation**: See [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) for technical details
- **Bug Reports**: [GitHub Issues](https://github.com/ArgusObscura/Lethe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArgusObscura/Lethe/discussions)
- **Email**: Contact via GitHub

---

## Acknowledgments

- **YOLOv8** by Ultralytics for state-of-the-art object detection
- **OpenCV** for comprehensive computer vision tools
- **FFmpeg** for reliable video codec support
- The autonomous driving research community for feedback and use cases

---

## Legal & Ethical Notes

Lethe is designed for **legitimate research and privacy protection**. Users are responsible for:

1. **Compliance**: Ensuring use complies with local privacy laws (GDPR, CCPA, etc.)
2. **Consent**: Having proper consent to process video footage
3. **Responsible Use**: Not using for surveillance evasion or harmful purposes
4. **Data Handling**: Securely storing both original and anonymized footage

---

**Remember**: *"The river Lethe makes the souls forget their past woes." — Virgil, Aeneid*

Transform your sensitive footage into privacy-respecting research data.
