"""The CLI must not hold its own copy of a config default."""

import click
import pytest

from anonymizer import cli as cli_module
from anonymizer.models.config import AnonymizationConfig, DetectionConfig


def option_default(command, name):
    for param in command.params:
        if name in param.opts:
            return param.default
    raise AssertionError(f"{command.name} has no {name}")


COMMANDS_WITH_CONFIDENCE = ["process", "stream", "batch", "detect"]


class TestNoShadowedDefaults:
    """A duplicated default silently wins, so changing the config does nothing.

    This is how confidence stayed at 0.5 after the config moved to 0.35: the
    CLI passed its own copy explicitly on every run.
    """

    @pytest.mark.parametrize("name", COMMANDS_WITH_CONFIDENCE)
    def test_confidence_has_no_cli_default(self, name):
        command = cli_module.cli.commands[name]

        assert option_default(command, "--confidence") is None

    def test_inference_size_has_no_cli_default(self):
        command = cli_module.cli.commands["process"]

        assert option_default(command, "--inference-size") is None

    def test_queue_confidence_has_no_cli_default(self):
        queue = cli_module.cli.commands["job"].commands["queue"]

        assert option_default(queue, "--confidence") is None


class TestConfigOwnsTheDefaults:
    def test_unset_confidence_uses_the_model_default(self):
        assert DetectionConfig().confidence_threshold == 0.35

    def test_unset_inference_size_stays_automatic(self):
        assert DetectionConfig().inference_size is None

    def test_explicit_values_still_apply(self):
        config = DetectionConfig(confidence_threshold=0.7, inference_size=800)

        assert config.confidence_threshold == 0.7
        assert config.inference_size == 800


class TestProcessBuildsExpectedConfig:
    def test_no_flags_yields_model_defaults(self, monkeypatch):
        """A bare `lethe process` must match AnonymizationConfig()."""
        captured = {}

        class FakePipeline:
            def __init__(self, config=None, **kw):
                captured["config"] = config
            def on(self, *a, **k): return self
            def process_video(self, *a, **k): return {}
            def cleanup(self): pass

        monkeypatch.setattr(cli_module, "AnonymizationPipeline", FakePipeline)

        from click.testing import CliRunner
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            CliRunner().invoke(
                cli_module.cli, ["process", path, "-o", "/tmp/out.mp4", "-q"]
            )
        finally:
            os.unlink(path)

        built = captured["config"]
        reference = AnonymizationConfig()
        assert built.face_config.confidence_threshold == reference.face_config.confidence_threshold
        assert built.face_config.inference_size == reference.face_config.inference_size
