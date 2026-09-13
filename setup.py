"""Setup configuration for Lethe anonymization tool."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lethe-anonymizer",
    version="0.1.0",
    author="ArgusObscura",
    description="A tool for anonymizing sensitive objects (faces, license plates) in vehicle camera videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArgusObscura/Lethe",
    project_urls={
        "Bug Tracker": "https://github.com/ArgusObscura/Lethe/issues",
        "Documentation": "https://github.com/ArgusObscura/Lethe/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lethe=anonymizer.cli:main",
        ],
    },
)
