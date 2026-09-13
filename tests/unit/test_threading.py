"""Importing Lethe must not silently pin torch to one core."""

import subprocess
import sys


def test_import_preserves_torch_thread_count():
    """ultralytics >= 8.3 sets torch to 1 thread on import.

    Left alone that makes CPU inference ~3.8x slower, and it is invisible:
    nothing errors, the run is just single-core. Checked in a subprocess
    because torch will not change its thread count once work has started.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; before = torch.get_num_threads();"
            " import anonymizer.models.loader;"
            " print(before, torch.get_num_threads())",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    before, after = map(int, result.stdout.split()[-2:])
    assert after >= before, f"torch threads dropped {before} -> {after} on import"
