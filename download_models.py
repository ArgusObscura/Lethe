#!/usr/bin/env python3
"""Download the detection models Lethe needs into its cache directory."""

import sys
from pathlib import Path
import urllib.request
import shutil


CACHE_DIR = Path.home() / ".cache" / "lethe" / "models"

MODELS = {
    "yolov8n-face.pt": (
        "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
        "face detection",
    ),
    "license-plate.pt": (
        "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
        "/resolve/main/license-plate-finetune-v1n.pt",
        "license plate detection",
    ),
}


def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    filled = int(50 * percent / 100)
    bar = "█" * filled + "░" * (50 - filled)
    sys.stdout.write(
        f"\r[{bar}] {percent:.1f}% "
        f"({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)"
    )
    sys.stdout.flush()


def download(filename: str, url: str, purpose: str) -> bool:
    """Fetch one model, skipping it if already cached."""
    model_path = CACHE_DIR / filename

    print(f"\n📦 {filename}  ({purpose})")

    if model_path.exists():
        print(f"   ✅ already present, {model_path.stat().st_size/(1024*1024):.1f} MB")
        return True

    print(f"   ⏳ downloading...")

    # Download to a temporary name so an interrupted transfer cannot leave a
    # truncated file that later looks cached.
    temp_path = CACHE_DIR / f"{filename}.tmp"
    try:
        urllib.request.urlretrieve(url, temp_path, progress)
        shutil.move(str(temp_path), str(model_path))
        print(f"\n   ✅ saved, {model_path.stat().st_size/(1024*1024):.1f} MB")
        return True
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        print(f"\n   ❌ failed: {e}")
        print(f"      Download {url}")
        print(f"      and save it as {model_path}")
        return False


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("📥 Lethe model downloader")
    print("=" * 70)
    print(f"📍 Cache directory: {CACHE_DIR}")

    results = {
        name: download(name, url, purpose)
        for name, (url, purpose) in MODELS.items()
    }

    print("\n" + "=" * 70)
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"❌ {len(failed)} of {len(results)} models unavailable: {', '.join(failed)}")
    else:
        print(f"✅ All {len(results)} models ready")
    print("=" * 70 + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
