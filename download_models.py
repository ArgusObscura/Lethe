#!/usr/bin/env python3
"""Download YOLOv8-face model to cache directory."""

import os
import sys
from pathlib import Path
import urllib.request
import shutil

def download_yolov8_face():
    """Download YOLOv8-face model."""

    cache_dir = Path.home() / ".cache" / "lethe" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "yolov8n-face.pt"

    print("\n" + "="*70)
    print("📥 YOLOv8-Face Model Downloader")
    print("="*70)
    print(f"📍 Cache directory: {cache_dir}")
    print(f"📦 Model path: {model_path}\n")

    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return True

    print("⏳ Downloading yolov8n-face.pt...")
    print("   This may take a minute or two...\n")

    # Download from GitHub releases
    url = "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"

    try:
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)")
            sys.stdout.flush()

        temp_path = cache_dir / "yolov8n-face.pt.tmp"
        urllib.request.urlretrieve(url, temp_path, download_progress)

        # Move to final location
        shutil.move(str(temp_path), str(model_path))

        print(f"\n\n✅ Model downloaded successfully!")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Location: {model_path}\n")
        return True

    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        print("\nTrying alternative download location...")

        # Try alternative
        alt_url = "https://huggingface.co/Bingsu/yolov8-face/resolve/main/yolov8n-face.pt"
        try:
            urllib.request.urlretrieve(alt_url, model_path, download_progress)
            print(f"\n\n✅ Model downloaded from alternative source!")
            return True
        except Exception as e2:
            print(f"\n\n❌ Both downloads failed: {e2}")
            print("\nFallback: Using standard YOLOv8n (less accurate for faces)")
            return False

if __name__ == "__main__":
    success = download_yolov8_face()
    sys.exit(0 if success else 1)
