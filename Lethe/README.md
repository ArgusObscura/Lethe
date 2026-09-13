# Lethe 🔮

**Lethe**: A comprehensive tool for anonymizing sensitive objects (faces, license plates) in vehicle camera videos for autonomous driving R&D.

Named after the Greek mythological river of forgetfulness, Lethe erases identifiable information from your research footage.

## Features ✨

- **Dual Object Detection**: 
  - 👤 Human face detection using YOLOv8-face
  - 📋 License plate detection using YOLOv8
  
- **Multiple Anonymization Techniques**:
  - 🔲 **Blur**: Gaussian blur for natural appearance
  - 📦 **Pixelate**: Block pixelation for stronger privacy
  - 🎨 **Mask**: Solid color replacement
  
- **Flexible Deployment**:
  - 💻 Command-line interface (CLI)
  - 📚 Python library for programmatic use
  - 🌐 REST API for distributed processing
  
- **Video Format Support**:
  - MP4 (H.264)
  - H.265 (HEVC)
  - Raw frame sequences (PNG/JPG)
  - Multiple resolutions (720p, 1080p, 4K)

## Installation

### Prerequisites
- Python 3.9+
- FFmpeg (for optimal codec support)
- CUDA-capable GPU (optional, for faster processing)

### From Source

```bash
# Clone the repository
git clone https://github.com/ArgusObscura/Lethe.git
cd Lethe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Lethe in development mode
pip install -e .
```

## Quick Start

### Using the CLI

```bash
# Basic usage - anonymize video with default settings
lethe process input_video.mp4 -o output_video.mp4

# With custom configuration
lethe process input_video.mp4 \
  -o output_video.mp4 \
  --method blur \
  --blur-kernel 31 \
  --confidence 0.5

# Process entire directory
lethe batch input_directory/ -o output_directory/
```

### Using the Python Library

```python
from anonymizer import AnonymizationPipeline, AnonymizationConfig

# Create configuration
config = AnonymizationConfig(
    method="blur",
    blur_kernel_size=31,
)

# Initialize pipeline
pipeline = AnonymizationPipeline(config)

# Process video
stats = pipeline.process_video("input.mp4", "output.mp4")

print(f"Processed {stats['frames_processed']} frames")
print(f"Detected {stats['faces_detected']} faces")
print(f"Detected {stats['license_plates_detected']} license plates")
```

### Using the REST API

```bash
# Start the API server
python -m anonymizer.api.server

# Upload and process video (in another terminal)
curl -X POST http://localhost:8000/anonymize \
  -F "video=@input.mp4" \
  -F "method=blur"
```

## Configuration

Lethe is highly configurable. Create a `config.yaml` file:

```yaml
anonymization:
  method: blur  # blur | pixelate | mask
  blur_kernel_size: 31
  pixelate_size: 15
  mask_color: [0, 0, 0]
  enable_face_detection: true
  enable_license_plate_detection: true

detection:
  face:
    confidence_threshold: 0.5
    device: cpu  # cpu | cuda
    batch_size: 8
  license_plate:
    confidence_threshold: 0.5
    device: cpu
    batch_size: 8

video:
  output_format: mp4  # mp4 | avi | mov | mkv
  crf: 23  # 0 (best) to 51 (worst)
  preserve_original: true
```

Then use it:

```bash
lethe process input.mp4 -o output.mp4 --config config.yaml
```

## Anonymization Methods

### Blur (Default) 🔲

Gaussian blur provides a natural appearance while effectively obscuring identity.

```python
config = AnonymizationConfig(
    method="blur",
    blur_kernel_size=31,  # Higher = more blur
)
```

**Best for**: Published papers, general R&D use

### Pixelation 📦

Block-based pixelation provides stronger privacy with more obvious anonymization.

```python
config = AnonymizationConfig(
    method="pixelate",
    pixelate_size=15,  # Larger = larger blocks
)
```

**Best for**: Strict privacy requirements, public datasets

### Masking 🎨

Solid color replacement removes all visual information from detected regions.

```python
config = AnonymizationConfig(
    method="mask",
    mask_color=(0, 0, 0),  # Black
)
```

**Best for**: Maximum privacy, sensitive footage

## Performance

### Detection Speed

- **Face Detection**: ~15-30 fps on modern GPU, ~2-5 fps on CPU
- **License Plate Detection**: ~10-20 fps on GPU, ~1-3 fps on CPU

### Memory Usage

- **Model Memory**: ~500MB (both models loaded)
- **Runtime Memory**: ~2-4GB depending on video resolution

### Recommendations

- Use GPU for production processing (10-100x faster)
- Process 1080p videos for optimal accuracy
- Batch processing for multiple videos

## Project Structure

```
Lethe/
├── src/anonymizer/
│   ├── core.py              # Main pipeline
│   ├── detector.py          # Object detection
│   ├── anonymizer.py        # Anonymization techniques
│   ├── video/
│   │   ├── reader.py        # Video input
│   │   ├── writer.py        # Video output
│   │   └── utils.py         # Utilities
│   ├── models/
│   │   ├── loader.py        # Model loading
│   │   └── config.py        # Configuration
│   ├── cli.py               # CLI interface
│   └── api/
│       ├── server.py        # REST API
│       └── routes.py        # API endpoints
├── tests/                   # Test suite
├── docs/                    # Documentation
└── examples/                # Example scripts
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/anonymizer

# Run specific test file
pytest tests/unit/test_anonymizer.py -v
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Roadmap

- [x] Phase 1: Core detection & anonymization
- [ ] Phase 2: CLI tool with batch processing
- [ ] Phase 3: Python library API
- [ ] Phase 4: REST API service
- [ ] Phase 5: Complete test suite & documentation
- [ ] Custom model fine-tuning support
- [ ] Real-time streaming anonymization
- [ ] Web UI for video processing
- [ ] Docker containerization

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Lethe in your research, please cite:

```bibtex
@software{lethe2024,
  title={Lethe: Vehicle Camera Video Anonymization Tool},
  author={ArgusObscura},
  url={https://github.com/ArgusObscura/Lethe},
  year={2024}
}
```

## Support

- 📖 [Documentation](docs/)
- 🐛 [Bug Reports](https://github.com/ArgusObscura/Lethe/issues)
- 💬 [Discussions](https://github.com/ArgusObscura/Lethe/discussions)

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for image processing
- Built for autonomous driving research and development

---

**Remember**: "The river Lethe makes the souls forget their past woes." — Virgil, Aeneid
