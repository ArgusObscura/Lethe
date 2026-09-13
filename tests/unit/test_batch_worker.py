"""Tests for the background job worker."""

import pytest

from anonymizer.batch import worker as worker_module
from anonymizer.batch.manager import BatchManager, JobStatus
from anonymizer.batch.worker import JobWorker


class FakePipeline:
    """Stands in for AnonymizationPipeline so tests need no models or video."""

    # Set by individual tests to steer behaviour.
    frames = 100
    raises = None
    instances = []

    def __init__(self, config=None):
        self.config = config
        self.cleaned_up = False
        FakePipeline.instances.append(self)

    def process_video(self, input_path, output_path, progress_callback=None):
        if FakePipeline.raises:
            raise FakePipeline.raises

        for frame in range(FakePipeline.frames):
            if progress_callback:
                progress_callback(frame, FakePipeline.frames)

        return {"frames_processed": FakePipeline.frames, "faces_detected": 3}

    def cleanup(self):
        self.cleaned_up = True


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    """Swap the real pipeline out for every test in this module."""
    FakePipeline.frames = 100
    FakePipeline.raises = None
    FakePipeline.instances = []
    monkeypatch.setattr(worker_module, "AnonymizationPipeline", FakePipeline)
    return FakePipeline


@pytest.fixture
def manager(tmp_path):
    return BatchManager(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def worker(manager):
    return JobWorker(manager=manager, check_every=10)


class TestRunOnce:
    def test_empty_queue_reports_no_work(self, worker):
        assert worker.run_once() is False

    def test_processes_a_queued_job(self, worker, manager):
        job = manager.queue_job("in.mp4", "out.mp4")

        assert worker.run_once() is True
        assert manager.get_job(job.job_id).status == JobStatus.COMPLETED

    def test_records_stats_from_pipeline(self, worker, manager):
        job = manager.queue_job("in.mp4", "out.mp4")
        worker.run_once()

        stats = manager.get_job(job.job_id).stats
        assert stats["frames_processed"] == 100
        assert stats["faces_detected"] == 3

    def test_drains_queue_in_order(self, worker, manager):
        first = manager.queue_job("a.mp4", "a_out.mp4")
        second = manager.queue_job("b.mp4", "b_out.mp4")

        worker.run_once()
        assert manager.get_job(first.job_id).status == JobStatus.COMPLETED
        assert manager.get_job(second.job_id).status == JobStatus.QUEUED

        worker.run_once()
        assert manager.get_job(second.job_id).status == JobStatus.COMPLETED

    def test_passes_stored_config_to_pipeline(self, worker, manager):
        from anonymizer.models.config import AnonymizationConfig

        manager.queue_job("in.mp4", "out.mp4", AnonymizationConfig(method="mask"))
        worker.run_once()

        assert FakePipeline.instances[0].config.method == "mask"


class TestProgress:
    def test_progress_is_persisted_during_processing(self, worker, manager):
        job = manager.queue_job("in.mp4", "out.mp4")
        worker.run_once()

        # Completion pins progress at 100 regardless of the last sample.
        assert manager.get_job(job.job_id).progress == 100.0

    def test_progress_written_only_every_nth_frame(self, manager, monkeypatch):
        writes = []
        original = manager.update_progress
        monkeypatch.setattr(
            manager,
            "update_progress",
            lambda jid, p: (writes.append(p), original(jid, p))[1],
        )

        manager.queue_job("in.mp4", "out.mp4")
        JobWorker(manager=manager, check_every=25).run_once()

        # 100 frames sampled every 25 -> frames 0, 25, 50, 75.
        assert len(writes) == 4


class TestFailure:
    def test_pipeline_error_marks_job_failed(self, worker, manager, fake_pipeline):
        fake_pipeline.raises = RuntimeError("codec exploded")
        job = manager.queue_job("in.mp4", "out.mp4")

        worker.run_once()

        stored = manager.get_job(job.job_id)
        assert stored.status == JobStatus.FAILED
        assert "codec exploded" in stored.error

    def test_failure_does_not_stop_the_worker(self, worker, manager, fake_pipeline):
        fake_pipeline.raises = RuntimeError("boom")
        manager.queue_job("a.mp4", "a_out.mp4")
        worker.run_once()

        fake_pipeline.raises = None
        second = manager.queue_job("b.mp4", "b_out.mp4")
        worker.run_once()

        assert manager.get_job(second.job_id).status == JobStatus.COMPLETED

    def test_pipeline_is_cleaned_up_after_failure(self, worker, manager, fake_pipeline):
        fake_pipeline.raises = RuntimeError("boom")
        manager.queue_job("in.mp4", "out.mp4")

        worker.run_once()

        assert FakePipeline.instances[0].cleaned_up is True


class TestCancellation:
    def test_running_job_stops_when_cancelled(self, manager):
        """Cancelling mid-run must unwind the frame loop, not run to completion."""
        job = manager.queue_job("in.mp4", "out.mp4")
        seen = []

        class CancellingPipeline(FakePipeline):
            def process_video(self, input_path, output_path, progress_callback=None):
                for frame in range(1000):
                    seen.append(frame)
                    if frame == 30:
                        manager.cancel_job(job.job_id)
                    progress_callback(frame, 1000)
                return {"frames_processed": 1000}

        worker = JobWorker(manager=manager, check_every=10)
        worker_module.AnonymizationPipeline = CancellingPipeline
        try:
            worker.run_once()
        finally:
            worker_module.AnonymizationPipeline = FakePipeline

        assert manager.get_job(job.job_id).status == JobStatus.CANCELLED
        # Stopped at the next multiple-of-10 boundary, nowhere near 1000.
        assert len(seen) < 100

    def test_cancelled_job_keeps_cancelled_status(self, worker, manager):
        job = manager.queue_job("in.mp4", "out.mp4")
        manager.cancel_job(job.job_id)

        # A cancelled job is no longer claimable, so nothing runs.
        assert worker.run_once() is False
        assert manager.get_job(job.job_id).status == JobStatus.CANCELLED
