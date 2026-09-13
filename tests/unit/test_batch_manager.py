"""Tests for the batch job queue."""

import pytest

from anonymizer.batch.manager import BatchManager, JobStatus
from anonymizer.models.config import AnonymizationConfig


@pytest.fixture
def manager(tmp_path):
    """A manager backed by a throwaway database."""
    return BatchManager(db_path=str(tmp_path / "jobs.db"))


class TestJobStatus:
    def test_terminal_states(self):
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal

    def test_active_states_are_not_terminal(self):
        assert not JobStatus.QUEUED.is_terminal
        assert not JobStatus.RUNNING.is_terminal


class TestQueueing:
    def test_queued_job_is_persisted(self, manager):
        job = manager.queue_job("in.mp4", "out.mp4")

        stored = manager.get_job(job.job_id)
        assert stored is not None
        assert stored.input_path == "in.mp4"
        assert stored.output_path == "out.mp4"
        assert stored.status == JobStatus.QUEUED

    def test_job_ids_are_unique(self, manager):
        ids = {manager.queue_job("in.mp4", "out.mp4").job_id for _ in range(20)}
        assert len(ids) == 20

    def test_config_round_trips(self, manager):
        config = AnonymizationConfig(method="pixelate", pixelate_size=42)
        job = manager.queue_job("in.mp4", "out.mp4", config)

        stored = manager.get_job(job.job_id)
        assert stored.config.method == "pixelate"
        assert stored.config.pixelate_size == 42

    def test_job_without_config_is_allowed(self, manager):
        job = manager.queue_job("in.mp4", "out.mp4")
        assert manager.get_job(job.job_id).config is None

    def test_missing_job_returns_none(self, manager):
        assert manager.get_job("nonexistent") is None

    def test_database_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "jobs.db")
        job_id = BatchManager(db_path=db).queue_job("in.mp4", "out.mp4").job_id

        assert BatchManager(db_path=db).get_job(job_id) is not None


class TestListing:
    def test_lists_newest_first(self, manager):
        first = manager.queue_job("a.mp4", "a_out.mp4")
        second = manager.queue_job("b.mp4", "b_out.mp4")

        listed = [j.job_id for j in manager.list_jobs()]
        assert listed.index(second.job_id) < listed.index(first.job_id)

    def test_filters_by_status(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        running = manager.claim_next_job()

        queued_only = manager.list_jobs(status=JobStatus.QUEUED)
        assert queued_only == []

        running_only = manager.list_jobs(status=JobStatus.RUNNING)
        assert [j.job_id for j in running_only] == [running.job_id]

    def test_respects_limit(self, manager):
        for i in range(5):
            manager.queue_job(f"{i}.mp4", f"{i}_out.mp4")

        assert len(manager.list_jobs(limit=2)) == 2

    def test_counts_by_status(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        manager.queue_job("b.mp4", "b_out.mp4")
        manager.claim_next_job()

        assert manager.counts_by_status() == {"queued": 1, "running": 1}


class TestClaiming:
    def test_claims_oldest_first(self, manager):
        first = manager.queue_job("a.mp4", "a_out.mp4")
        manager.queue_job("b.mp4", "b_out.mp4")

        assert manager.claim_next_job().job_id == first.job_id

    def test_claim_marks_running_and_stamps_start(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        claimed = manager.claim_next_job()

        assert claimed.status == JobStatus.RUNNING
        assert claimed.started_at is not None

    def test_a_job_is_claimed_only_once(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")

        assert manager.claim_next_job() is not None
        assert manager.claim_next_job() is None

    def test_empty_queue_returns_none(self, manager):
        assert manager.claim_next_job() is None

    def test_concurrent_workers_never_share_a_job(self, tmp_path):
        """Each job must go to exactly one worker."""
        import threading

        db = str(tmp_path / "jobs.db")
        manager = BatchManager(db_path=db)
        for i in range(12):
            manager.queue_job(f"{i}.mp4", f"{i}_out.mp4")

        claimed = []
        lock = threading.Lock()

        def drain():
            worker_view = BatchManager(db_path=db)
            while True:
                job = worker_view.claim_next_job()
                if job is None:
                    return
                with lock:
                    claimed.append(job.job_id)

        threads = [threading.Thread(target=drain) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(claimed) == 12
        assert len(set(claimed)) == 12


class TestCompletion:
    def test_complete_stores_stats(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        manager.claim_next_job()
        manager.complete_job(job.job_id, {"frames_processed": 100, "faces_detected": 7})

        stored = manager.get_job(job.job_id)
        assert stored.status == JobStatus.COMPLETED
        assert stored.progress == 100.0
        assert stored.stats["faces_detected"] == 7
        assert stored.completed_at is not None

    def test_fail_stores_error(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        manager.fail_job(job.job_id, "codec exploded")

        stored = manager.get_job(job.job_id)
        assert stored.status == JobStatus.FAILED
        assert stored.error == "codec exploded"

    def test_update_progress(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        manager.update_progress(job.job_id, 42.5)

        assert manager.get_job(job.job_id).progress == 42.5

    def test_duration_measured_from_start(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        claimed = manager.claim_next_job()
        manager.complete_job(claimed.job_id, {})

        assert manager.get_job(claimed.job_id).duration_seconds >= 0

    def test_duration_is_none_before_start(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        assert manager.get_job(job.job_id).duration_seconds is None


class TestCancellation:
    def test_cancel_queued_job(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")

        assert manager.cancel_job(job.job_id) is True
        assert manager.get_job(job.job_id).status == JobStatus.CANCELLED

    def test_cancelled_job_is_never_claimed(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        manager.cancel_job(job.job_id)

        assert manager.claim_next_job() is None

    def test_cancel_running_job(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        claimed = manager.claim_next_job()

        assert manager.cancel_job(claimed.job_id) is True
        assert manager.is_cancelled(claimed.job_id)

    def test_cannot_cancel_completed_job(self, manager):
        job = manager.queue_job("a.mp4", "a_out.mp4")
        manager.complete_job(job.job_id, {})

        assert manager.cancel_job(job.job_id) is False
        assert manager.get_job(job.job_id).status == JobStatus.COMPLETED

    def test_cancel_missing_job_reports_failure(self, manager):
        assert manager.cancel_job("nonexistent") is False

    def test_is_cancelled_false_for_running(self, manager):
        manager.queue_job("a.mp4", "a_out.mp4")
        claimed = manager.claim_next_job()

        assert manager.is_cancelled(claimed.job_id) is False
