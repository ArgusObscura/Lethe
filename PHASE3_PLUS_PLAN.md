# Phase 3+: Real-Time Monitoring & Streaming - Comprehensive Plan

## Overview

Phase 3+ extends the Phase 3 event system with real-time monitoring, streaming support, and batch job management.

**Duration**: 2-3 weeks
**Priority**: High (enables production workflows)
**Status**: Planning

---

## Part 1: CLI Event Integration (Week 1)

### Goal
Integrate event system into CLI for real-time progress monitoring during video processing.

### Architecture

```
lethe process video.mp4
    ↓
[CLI Command Handler]
    ↓
[AnonymizationPipeline]
    ↓ (emits events)
[EventEmitter]
    ↓
[CLI Progress Handler]
    ├→ [Progress Bar Display]
    ├→ [Live Statistics]
    ├→ [Event Log File]
    └→ [Console Output]
```

### Implementation Tasks

#### 1.1 Create `cli/progress.py` (NEW)
**Purpose**: Handle real-time progress display and logging

**Classes**:
```python
class ProgressDisplay:
    """Real-time progress bar and statistics."""
    - __init__(total_frames, output_file=None)
    - on_frame_completed(event) → updates progress bar
    - on_detection(event) → updates detection counts
    - get_summary() → returns final statistics
    - print_summary() → displays final statistics

class EventLogger:
    """Log events to file with timestamps."""
    - __init__(log_file)
    - log(event) → writes event to file
    - on_pipeline_event(event) → handles all events
```

**Dependencies**:
- `tqdm` (progress bar) - already installed
- `datetime` (timestamps)
- `json` (structured logging)

#### 1.2 Update `cli.py`
**File**: `src/anonymizer/cli.py`

**Changes**:
```python
# In process_command():

# Create progress display
progress = ProgressDisplay(
    total_frames=video_info.frame_count,
    output_file="processing.log" if args.log else None
)

# Register event callbacks
pipeline.on(EventType.VIDEO_OPENED, progress.on_video_opened)
pipeline.on(EventType.FRAME_COMPLETED, progress.on_frame_completed)
pipeline.on(EventType.FACES_DETECTED, progress.on_detection)
pipeline.on(EventType.PLATES_DETECTED, progress.on_detection)
pipeline.on(EventType.PIPELINE_COMPLETED, progress.on_complete)

# Process
pipeline.process_video(input_video, output_video)

# Show summary
progress.print_summary()
```

**New CLI Options**:
```bash
lethe process input.mp4 -o output.mp4 \
    --log                    # Save event log
    --log-file custom.log    # Custom log file
    --no-progress            # Disable progress bar
    --quiet                  # Minimal output
```

#### 1.3 Update `batch.py`
**File**: `src/anonymizer/cli.py` (batch_command function)

**Changes**:
- Show progress for entire batch
- Track per-video progress
- Aggregate statistics across videos
- Generate batch report

```python
class BatchProgress:
    """Track batch processing progress."""
    - total_videos: int
    - videos_completed: int
    - total_detections: int
    - total_time: float
```

#### 1.4 Create Log File Format
**File**: `logs/processing_2026-09-13_180000.log`

```json
{
    "timestamp": "2026-09-13T18:00:00",
    "video": "test.mov",
    "status": "processing",
    "events": [
        {
            "type": "pipeline_started",
            "timestamp": "2026-09-13T18:00:00",
            "data": {}
        },
        {
            "type": "frame_completed",
            "timestamp": "2026-09-13T18:00:01",
            "frame": 1,
            "progress": 0.2,
            "fps": 210
        },
        {
            "type": "faces_detected",
            "timestamp": "2026-09-13T18:00:01",
            "frame": 1,
            "count": 2,
            "total": 42
        }
    ],
    "summary": {
        "total_frames": 450,
        "faces": 801,
        "plates": 1385,
        "duration": 120.5,
        "fps": 3.7
    }
}
```

### Testing

**Unit Tests** (`tests/unit/test_cli_progress.py`):
- ProgressDisplay creation
- Event callback invocation
- Log file writing
- Summary generation

**Integration Tests** (`tests/integration/test_cli_events.py`):
- Full CLI flow with progress display
- Batch processing with progress
- Log file content validation
- No regression in core functionality

### Deliverables

- ✅ CLI progress display during processing
- ✅ Event logging to file
- ✅ Batch statistics aggregation
- ✅ Tests for new functionality
- ✅ Updated README with examples

---

## Part 2: Streaming Support (Week 2)

### Goal
Enable processing of frame streams, not just video files.

### Implementation Tasks

#### 2.1 Create `video/stream.py` (NEW)
**Purpose**: Handle frame stream input/output

**Classes**:
```python
class FrameStream:
    """Abstract frame stream interface."""
    - read_frame() → (frame, metadata)
    - write_frame(frame) → None
    - close() → None

class WebcamStream(FrameStream):
    """Live webcam stream."""
    - read from OpenCV capture

class HTTPStream(FrameStream):
    """HTTP stream (MJPEG, etc)."""
    - read from HTTP endpoint

class RTSPStream(FrameStream):
    """RTSP stream."""
    - read from RTSP source
```

#### 2.2 Update Pipeline
**File**: `src/anonymizer/core.py`

```python
def process_stream(self, input_stream, output_stream, max_frames=None):
    """Process frame stream with events."""
    - Emit STREAM_STARTED event
    - Loop through frames
    - Emit per-frame events
    - Handle stream end
    - Emit STREAM_COMPLETED event
```

#### 2.3 CLI Streaming Command
**File**: `src/anonymizer/cli.py`

```bash
lethe stream --input webcam://0 --output output.mp4
lethe stream --input rtsp://camera.local/stream --output output.mp4
lethe stream --input http://localhost:8000/mjpeg --output output.mp4
```

### Testing

- Unit tests for each stream type
- Integration tests for pipeline
- Real-time performance tests

### Deliverables

- ✅ Frame stream abstraction
- ✅ Multiple stream input support
- ✅ CLI streaming commands
- ✅ Tests

---

## Part 3: Batch Job Management (Week 3)

### Goal
Track and manage multiple video processing jobs.

### Implementation Tasks

#### 3.1 Create `batch/manager.py` (NEW)
**Purpose**: Job queue and tracking

**Classes**:
```python
class BatchJob:
    """Single job metadata."""
    - job_id: str
    - status: "queued" | "processing" | "completed" | "failed"
    - progress: float
    - created_at: datetime
    - started_at: datetime
    - completed_at: datetime
    - stats: Dict

class BatchManager:
    """Manage multiple jobs."""
    - queue_job(input, output, config)
    - get_job_status(job_id)
    - get_all_jobs()
    - cancel_job(job_id)
```

#### 3.2 Job Database
**File**: `~/.cache/lethe/jobs.db`

SQLite database:
```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    input_path TEXT,
    output_path TEXT,
    status TEXT,
    progress REAL,
    created_at DATETIME,
    started_at DATETIME,
    completed_at DATETIME,
    stats JSON
);
```

#### 3.3 CLI Job Commands
**File**: `src/anonymizer/cli.py`

```bash
lethe job queue input.mp4 -o output.mp4
lethe job status <job_id>
lethe job list
lethe job cancel <job_id>
lethe job results <job_id>
```

#### 3.4 Background Job Worker
**File**: `src/anonymizer/batch/worker.py`

```python
class JobWorker:
    """Process jobs from queue."""
    - run_forever()  # Main loop
    - process_next_job()
    - emit_job_events()
```

### Testing

- Unit tests for job tracking
- Database operations
- CLI commands
- Job persistence

### Deliverables

- ✅ Job queue system
- ✅ Job database/persistence
- ✅ CLI job commands
- ✅ Background worker
- ✅ Tests

---

## Architecture Summary

### File Structure
```
src/anonymizer/
├── cli.py                    [UPDATED]
├── cli/
│   ├── __init__.py
│   └── progress.py           [NEW]
├── batch/
│   ├── __init__.py
│   ├── manager.py            [NEW]
│   └── worker.py             [NEW]
├── video/
│   ├── stream.py             [NEW]
│   ├── reader.py             [EXISTS]
│   └── writer.py             [EXISTS]

tests/
├── unit/
│   ├── test_cli_progress.py  [NEW]
│   ├── test_stream.py        [NEW]
│   ├── test_batch_manager.py [NEW]
│
├── integration/
│   ├── test_cli_events.py    [NEW]
│   ├── test_streaming.py     [NEW]
│   └── test_batch_worker.py  [NEW]
```

### Dependencies (All installed)
- `tqdm` - Progress bars
- `loguru` - Logging
- `click` - CLI (exists)
- `ultralytics` - Detection (exists)
- `opencv-python` - Video I/O (exists)
- `sqlalchemy` - ORM for jobs (NEW - optional)

---

## Implementation Approach

### Phase 3+.1: CLI Events (Highest Priority)
1. Create `cli/progress.py`
2. Update `cli.py` with event integration
3. Add tests
4. **Deliverable**: Real-time progress in CLI ✅

### Phase 3+.2: Streaming (Medium Priority)
1. Create `video/stream.py`
2. Add stream support to pipeline
3. Add CLI streaming commands
4. **Deliverable**: Support for camera/RTSP/HTTP streams ✅

### Phase 3+.3: Batch Management (Lower Priority)
1. Create batch manager and worker
2. Implement job database
3. Add CLI job commands
4. **Deliverable**: Job queue and tracking ✅

---

## Success Criteria

### Part 1 Complete When:
- ✅ `lethe process` shows real-time progress bar
- ✅ Event log file is created and valid
- ✅ Batch processing shows aggregate stats
- ✅ All tests pass
- ✅ No performance regression

### Part 2 Complete When:
- ✅ Can process webcam stream
- ✅ Can process RTSP stream
- ✅ Can process HTTP stream
- ✅ All tests pass
- ✅ CLI commands work

### Part 3 Complete When:
- ✅ Jobs can be queued
- ✅ Job status trackable
- ✅ Job history persists
- ✅ Background worker processes jobs
- ✅ All tests pass

---

## Timeline Estimate

| Part | Effort | Timeline |
|------|--------|----------|
| 1: CLI Events | 8-10 hours | 1-2 days |
| 2: Streaming | 10-12 hours | 2-3 days |
| 3: Batch Jobs | 12-16 hours | 3-4 days |
| **Total** | **30-38 hours** | **1-2 weeks** |

---

## Risk Mitigation

### Risk 1: Event system integration breaks existing code
**Mitigation**: Comprehensive integration tests, backward compatibility checks

### Risk 2: Real-time updates slow down processing
**Mitigation**: Async event callbacks, benchmark performance impact

### Risk 3: Streaming introduces memory leaks
**Mitigation**: Stream cleanup tests, resource monitoring

### Risk 4: Job database locks cause issues
**Mitigation**: SQLite with WAL mode, transaction testing

---

## Notes

- Event system (Phase 3) is already complete and tested ✅
- All dependencies are already installed ✅
- Can start Phase 3+.1 immediately ✅
- Streaming support (3+.2) can be optional MVP
- Batch management (3+.3) can be released as Phase 4

---

## Questions for Approval

1. Should we start with Part 1 (CLI Events) immediately?
2. Do you want Parts 2 & 3, or focus on Part 1 first?
3. Any specific priorities or constraints?

