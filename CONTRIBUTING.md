# Contributing to Lethe

First off, thank you for considering contributing to Lethe! It's people like you that make Lethe such a great tool for autonomous driving research and privacy protection.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**
* **Include your environment details** (OS, Python version, GPU/CPU, etc.)

### ✨ Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### 🔧 Pull Requests

* Fill in the required template
* Follow the Python styleguides
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- FFmpeg
- GPU (optional, for faster development)

### Setting Up Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Lethe.git
cd Lethe
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies in development mode**

```bash
pip install -r requirements.txt
pip install -e .
```

4. **Install development tools**

```bash
pip install black flake8 isort mypy pytest-cov
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/anonymizer --cov-report=html

# Run specific test file
pytest tests/unit/test_anonymizer.py -v

# Run with output
pytest tests/ -v -s
```

### Code Style

We follow PEP 8 with these tools:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Git Workflow

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

- Write clean, well-documented code
- Add type hints
- Write docstrings for new functions
- Add tests for new functionality

3. **Run tests and linting**

```bash
pytest tests/ -v
black src/ tests/
flake8 src/ tests/
mypy src/
```

4. **Commit your changes**

```bash
git add .
git commit -m "Clear description of your changes"
```

5. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**

---

## Styleguides

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use f-strings for string formatting

### Docstring Style

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add face detection confidence threshold option

- Allow users to customize face detection sensitivity
- Fixes #123
- Implements feature request from #456
```

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Test additions

---

## Project Structure

```
Lethe/
├── src/anonymizer/           # Main package
│   ├── core.py              # Core pipeline
│   ├── detector.py          # Detection logic
│   ├── anonymizer.py        # Anonymization methods
│   ├── cli.py               # CLI interface
│   ├── video/               # Video I/O
│   ├── models/              # Model management
│   └── api/                 # REST API (future)
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py          # Fixtures
└── docs/                    # Documentation
```

### Key Files to Know

- **src/anonymizer/core.py** - Main AnonymizationPipeline class
- **src/anonymizer/detector.py** - ObjectDetector and Detection classes
- **src/anonymizer/models/config.py** - Configuration models
- **tests/unit/test_anonymizer.py** - Anonymization tests
- **tests/unit/test_video.py** - Video I/O tests

---

## Areas Needing Help

We especially welcome contributions in these areas:

- [ ] **Custom License Plate Models** - Fine-tune LP detection for different regions
- [ ] **Additional Anonymization Techniques** - Synthetic face replacement, voice anonymization
- [ ] **Performance Optimization** - GPU acceleration, batch optimization
- [ ] **Documentation** - API docs, tutorials, best practices
- [ ] **Real-World Datasets** - Test videos and benchmarks
- [ ] **REST API Implementation** - Phase 4 completion
- [ ] **Web UI** - User-friendly interface
- [ ] **CI/CD Pipeline** - GitHub Actions setup

---

## Release Process

1. Update version in `setup.py` and `src/anonymizer/__init__.py`
2. Update `CHANGELOG.md` with changes
3. Create a commit: `git commit -m "Release v0.2.0"`
4. Create a tag: `git tag v0.2.0`
5. Push to GitHub: `git push origin main --tags`
6. Create GitHub Release with changelog

---

## Additional Notes

### Issue and Pull Request Labels

* `bug` - Something isn't working
* `enhancement` - New feature or request
* `documentation` - Improvements or additions to documentation
* `good first issue` - Good for newcomers
* `help wanted` - Extra attention is needed
* `question` - Further information is requested

### Testing Philosophy

We aim for >80% test coverage. When adding features:

1. Write tests BEFORE writing code (TDD approach)
2. Test both happy paths and error cases
3. Include integration tests for new commands/APIs
4. Test with different configurations

### Documentation Philosophy

Code should be self-documenting, but also:

1. Add docstrings to all public functions
2. Update README if adding public features
3. Add examples for new functionality
4. Keep documentation in sync with code

---

## Questions?

Feel free to ask questions:

- 💬 [GitHub Discussions](https://github.com/ArgusObscura/Lethe/discussions)
- 🐛 [GitHub Issues](https://github.com/ArgusObscura/Lethe/issues)

---

## License

By contributing to Lethe, you agree that your contributions will be licensed under its MIT License.

Thank you for contributing! 🎉
