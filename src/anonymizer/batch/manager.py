"""Job queue and persistence for batch anonymization."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

from ..models.config import AnonymizationConfig


DEFAULT_DB_PATH = Path.home() / ".cache" / "lethe" / "jobs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    config TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    stats TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
"""


class JobStatus(str, Enum):
    """Lifecycle states of a batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is possible."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass
class BatchJob:
    """A single queued anonymization job."""

    job_id: str
    input_path: str
    output_path: str
    status: JobStatus
    created_at: datetime
    progress: float = 0.0
    config: Optional[AnonymizationConfig] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Wall-clock processing time, if the job has run."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "BatchJob":
        """Build a job from a database row."""
        config = None
        if row["config"]:
            config = AnonymizationConfig.model_validate_json(row["config"])

        def parse_time(value: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(value) if value else None

        return cls(
            job_id=row["job_id"],
            input_path=row["input_path"],
            output_path=row["output_path"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            config=config,
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=parse_time(row["started_at"]),
            completed_at=parse_time(row["completed_at"]),
            stats=json.loads(row["stats"]) if row["stats"] else {},
            error=row["error"],
        )


class BatchManager:
    """Persistent job queue backed by SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        """Open (and create if needed) the job database.

        Args:
            db_path: Database location. Defaults to ``~/.cache/lethe/jobs.db``.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            # WAL lets the worker write progress while the CLI reads status.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

        logger.info(f"Job database: {self.db_path}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and closes on exit."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def queue_job(
        self,
        input_path: str,
        output_path: str,
        config: Optional[AnonymizationConfig] = None,
    ) -> BatchJob:
        """Add a job to the queue.

        Returns:
            The newly created job.
        """
        job = BatchJob(
            job_id=uuid.uuid4().hex[:12],
            input_path=str(input_path),
            output_path=str(output_path),
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            config=config,
        )

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (job_id, input_path, output_path, status, progress, config, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id,
                    job.input_path,
                    job.output_path,
                    job.status.value,
                    job.progress,
                    config.model_dump_json() if config else None,
                    job.created_at.isoformat(),
                ),
            )

        logger.info(f"Queued job {job.job_id}: {input_path}")
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Look up a single job, or None if it does not exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        return BatchJob.from_row(row) if row else None

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[BatchJob]:
        """List jobs newest first, optionally filtered by status."""
        query = "SELECT * FROM jobs"
        params: list = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [BatchJob.from_row(row) for row in rows]

    def claim_next_job(self) -> Optional[BatchJob]:
        """Atomically take the oldest queued job and mark it running.

        The conditional UPDATE is what makes this safe to call from more than
        one worker: only the worker whose UPDATE actually changes a row owns
        the job.

        Returns:
            The claimed job, or None if the queue is empty.
        """
        while True:
            with self._connect() as conn:
                row = conn.execute(
                    """SELECT job_id FROM jobs WHERE status = ?
                       ORDER BY created_at LIMIT 1""",
                    (JobStatus.QUEUED.value,),
                ).fetchone()

                if not row:
                    return None

                started = datetime.now().isoformat()
                cursor = conn.execute(
                    """UPDATE jobs SET status = ?, started_at = ?
                       WHERE job_id = ? AND status = ?""",
                    (
                        JobStatus.RUNNING.value,
                        started,
                        row["job_id"],
                        JobStatus.QUEUED.value,
                    ),
                )

                if cursor.rowcount == 1:
                    claimed = conn.execute(
                        "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
                    ).fetchone()
                    logger.info(f"Claimed job {row['job_id']}")
                    return BatchJob.from_row(claimed)

            # Another worker won the race; try the next candidate.

    def update_progress(self, job_id: str, progress: float) -> None:
        """Record how far along a running job is (0-100)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ? WHERE job_id = ?",
                (progress, job_id),
            )

    def complete_job(self, job_id: str, stats: Dict[str, Any]) -> None:
        """Mark a job finished and store its statistics."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE jobs SET status = ?, progress = 100.0,
                   completed_at = ?, stats = ? WHERE job_id = ?""",
                (
                    JobStatus.COMPLETED.value,
                    datetime.now().isoformat(),
                    json.dumps(stats),
                    job_id,
                ),
            )
        logger.info(f"Completed job {job_id}")

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job failed with the reason."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE jobs SET status = ?, completed_at = ?, error = ?
                   WHERE job_id = ?""",
                (
                    JobStatus.FAILED.value,
                    datetime.now().isoformat(),
                    error,
                    job_id,
                ),
            )
        logger.error(f"Failed job {job_id}: {error}")

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job.

        A queued job stops before it starts. A running job is not killed here:
        the worker notices the status change and stops at the next frame
        boundary, so the partial output is closed cleanly.

        Returns:
            True if the job was cancelled, False if it was missing or already
            in a terminal state.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET status = ?, completed_at = ?
                   WHERE job_id = ? AND status IN (?, ?)""",
                (
                    JobStatus.CANCELLED.value,
                    datetime.now().isoformat(),
                    job_id,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )
            cancelled = cursor.rowcount == 1

        if cancelled:
            logger.info(f"Cancelled job {job_id}")
        return cancelled

    def is_cancelled(self, job_id: str) -> bool:
        """Whether a cancellation has been requested for this job."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        return bool(row) and row["status"] == JobStatus.CANCELLED.value

    def counts_by_status(self) -> Dict[str, int]:
        """Number of jobs in each status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()

        return {row["status"]: row["n"] for row in rows}
