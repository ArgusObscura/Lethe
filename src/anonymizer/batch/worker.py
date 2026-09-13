"""Background worker that drains the job queue."""

import time
from typing import Optional

from loguru import logger

from ..core import AnonymizationPipeline
from .manager import BatchJob, BatchManager, JobStatus


class JobCancelled(Exception):
    """Raised inside the frame loop to unwind a cancelled job."""


class JobWorker:
    """Claims queued jobs and processes them one at a time."""

    def __init__(
        self,
        manager: Optional[BatchManager] = None,
        poll_interval: float = 2.0,
        check_every: int = 30,
    ):
        """Create a worker.

        Args:
            manager: Job store to draw from. Defaults to a new BatchManager.
            poll_interval: Seconds to wait when the queue is empty.
            check_every: How many frames between database writes. Persisting
                progress on every frame would dominate processing time.
        """
        self.manager = manager or BatchManager()
        self.poll_interval = poll_interval
        self.check_every = check_every

    def run_once(self) -> bool:
        """Claim and process a single job.

        Returns:
            True if a job was processed, False if the queue was empty.
        """
        job = self.manager.claim_next_job()

        if job is None:
            return False

        self._process(job)
        return True

    def run_forever(self) -> None:
        """Process jobs until interrupted, sleeping when the queue is empty."""
        logger.info("Worker started, waiting for jobs...")

        try:
            while True:
                if not self.run_once():
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker stopped")

    def _process(self, job: BatchJob) -> None:
        """Run one job to completion, recording the outcome."""
        logger.info(f"Processing job {job.job_id}: {job.input_path}")

        pipeline = AnonymizationPipeline(job.config)

        def on_progress(frame_number: int, total_frames: int) -> None:
            # Called directly by the pipeline, unlike event callbacks whose
            # exceptions the emitter swallows, so raising here actually stops
            # the frame loop.
            if frame_number % self.check_every:
                return

            if self.manager.is_cancelled(job.job_id):
                raise JobCancelled()

            if total_frames:
                self.manager.update_progress(
                    job.job_id, (frame_number / total_frames) * 100
                )

        try:
            stats = pipeline.process_video(
                job.input_path,
                job.output_path,
                progress_callback=on_progress,
            )
            self.manager.complete_job(job.job_id, stats)

        except JobCancelled:
            # cancel_job already set the status; the partial output was closed
            # cleanly on the way out.
            logger.info(f"Job {job.job_id} cancelled, stopped at frame boundary")

        except Exception as e:
            self.manager.fail_job(job.job_id, str(e))

        finally:
            pipeline.cleanup()
