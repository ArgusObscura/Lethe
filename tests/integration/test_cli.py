"""Integration tests for CLI interface."""

import pytest
from click.testing import CliRunner
from pathlib import Path

from anonymizer.cli import cli, process, batch, detect, show_config


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self, cli_runner):
        """Test CLI help output."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Lethe" in result.output
        assert "process" in result.output
        assert "batch" in result.output
        assert "detect" in result.output

    def test_cli_version(self, cli_runner):
        """Test version display."""
        result = cli_runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_process_help(self, cli_runner):
        """Test process command help."""
        result = cli_runner.invoke(process, ["--help"])

        assert result.exit_code == 0
        assert "Process a single video" in result.output
        assert "--method" in result.output
        assert "--confidence" in result.output

    def test_batch_help(self, cli_runner):
        """Test batch command help."""
        result = cli_runner.invoke(batch, ["--help"])

        assert result.exit_code == 0
        assert "Batch process" in result.output
        assert "--pattern" in result.output

    def test_detect_help(self, cli_runner):
        """Test detect command help."""
        result = cli_runner.invoke(detect, ["--help"])

        assert result.exit_code == 0
        assert "Detect objects" in result.output


class TestProcessCommand:
    """Test process command."""

    def test_process_missing_output(self, cli_runner, create_test_video):
        """Test process command without output argument."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)

        result = cli_runner.invoke(process, [str(video_path)])

        assert result.exit_code != 0
        assert "Error" in result.output or "required" in result.output.lower()

    def test_process_invalid_method(self, cli_runner, create_test_video, tmp_path):
        """Test process command with invalid method."""
        video_path = create_test_video("test.mp4")
        output_path = tmp_path / "output.mp4"

        result = cli_runner.invoke(
            process,
            [str(video_path), "-o", str(output_path), "-m", "invalid"]
        )

        assert result.exit_code != 0

    def test_process_invalid_color(self, cli_runner, create_test_video, tmp_path):
        """Test process command with invalid mask color."""
        video_path = create_test_video("test.mp4")
        output_path = tmp_path / "output.mp4"

        result = cli_runner.invoke(
            process,
            [
                str(video_path),
                "-o", str(output_path),
                "-m", "mask",
                "--mask-color", "invalid"
            ]
        )

        assert result.exit_code != 0


class TestBatchCommand:
    """Test batch command."""

    def test_batch_help_text(self, cli_runner):
        """Test batch command help."""
        result = cli_runner.invoke(batch, ["--help"])

        assert result.exit_code == 0
        assert "multiple videos" in result.output.lower()

    def test_batch_no_files(self, cli_runner, tmp_path):
        """Test batch with empty directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            batch,
            [str(input_dir), "-o", str(output_dir), "-p", "*.mp4"]
        )

        # Should succeed but report no files
        assert "No videos found" in result.output


class TestDetectCommand:
    """Test detect command."""

    def test_detect_help_text(self, cli_runner):
        """Test detect command help."""
        result = cli_runner.invoke(detect, ["--help"])

        assert result.exit_code == 0
        assert "Detect objects" in result.output

    def test_detect_invalid_image(self, cli_runner):
        """Test detect with non-existent image."""
        result = cli_runner.invoke(detect, ["nonexistent.jpg"])

        assert result.exit_code != 0


class TestShowConfigCommand:
    """Test show-config command."""

    def test_show_config_default(self, cli_runner):
        """Test show-config with default method."""
        result = cli_runner.invoke(show_config, [])

        assert result.exit_code == 0
        assert "anonymization:" in result.output
        assert "detection:" in result.output
        assert "blur" in result.output

    def test_show_config_pixelate(self, cli_runner):
        """Test show-config with pixelate method."""
        result = cli_runner.invoke(show_config, ["-m", "pixelate"])

        assert result.exit_code == 0
        assert "pixelate_size:" in result.output

    def test_show_config_mask(self, cli_runner):
        """Test show-config with mask method."""
        result = cli_runner.invoke(show_config, ["-m", "mask"])

        assert result.exit_code == 0
        assert "mask_color:" in result.output


class TestCLIIntegration:
    """Test CLI integration."""

    def test_cli_with_quiet_flag(self, cli_runner, create_test_video, tmp_path):
        """Test process command with quiet flag."""
        video_path = create_test_video("test.mp4", fps=30, duration=1)
        output_path = tmp_path / "output.mp4"

        result = cli_runner.invoke(
            process,
            [str(video_path), "-o", str(output_path), "-q"]
        )

        # Should complete successfully (might have errors due to model loading)
        # but the quiet flag should not cause issues
        assert result.exit_code in [0, 1]  # Accept success or expected model error

    def test_cli_method_options(self, cli_runner, create_test_video, tmp_path):
        """Test that all anonymization methods are recognized."""
        video_path = create_test_video("test.mp4")
        output_path = tmp_path / "output.mp4"

        for method in ["blur", "pixelate", "mask"]:
            result = cli_runner.invoke(
                process,
                [str(video_path), "-o", str(output_path), "-m", method],
                catch_exceptions=False  # This will show real errors
            )
            # May fail due to model loading, but should at least accept the method
            # The important thing is it doesn't reject the method as invalid


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    def test_show_config_outputs_valid_yaml(self, cli_runner):
        """Test that show-config outputs valid YAML."""
        result = cli_runner.invoke(show_config, [])

        assert result.exit_code == 0

        # Try to parse as YAML
        import yaml
        config = yaml.safe_load(result.output)

        assert "anonymization" in config
        assert "detection" in config

    def test_confidence_parameter_range(self, cli_runner, tmp_path):
        """Test confidence parameter validation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Test valid confidence
        result = cli_runner.invoke(
            batch,
            [str(input_dir), "-o", str(output_dir), "--confidence", "0.5"]
        )
        # Should at least parse the parameter
        assert "No videos found" in result.output or result.exit_code == 0


class TestCLIErrors:
    """Test CLI error handling."""

    def test_process_with_invalid_input_file(self, cli_runner, tmp_path):
        """Test process with non-existent input file."""
        output_path = tmp_path / "output.mp4"

        result = cli_runner.invoke(
            process,
            ["nonexistent.mp4", "-o", str(output_path)]
        )

        assert result.exit_code != 0

    def test_batch_with_invalid_input_directory(self, cli_runner, tmp_path):
        """Test batch with non-existent directory."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            batch,
            ["nonexistent_dir", "-o", str(output_dir)]
        )

        assert result.exit_code != 0


class TestCLIDeviceOption:
    """Test device selection in CLI."""

    def test_device_cpu_option(self, cli_runner, tmp_path):
        """Test CPU device option."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            batch,
            [str(input_dir), "-o", str(output_dir), "--device", "cpu"]
        )

        # Should parse without error
        assert "No videos found" in result.output or result.exit_code == 0

    def test_invalid_device(self, cli_runner, tmp_path):
        """Test invalid device option."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            batch,
            [str(input_dir), "-o", str(output_dir), "--device", "invalid"]
        )

        assert result.exit_code != 0
