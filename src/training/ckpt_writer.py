"""Background checkpoint writer: torch.save off the training path.

Rank 0 writes the epoch's weights and trainer state every epoch; written
inline, a slow shared filesystem stalls rank 0 and, through the next DDP
all-reduce, every other rank. The GPU->CPU snapshot stays synchronous (the
next optimizer step overwrites the parameters); only torch.save and the
rename move to a background thread.

One consumer thread, not a pool: find_latest_trainer_state pairs
full_epoch{N}.pt with epoch_weights/epoch_{N}.pt, so the weights must land
before the trainer state that points at them. A FIFO consumer preserves
submission order, and with each write's own tmp + rename the files on disk
are always a prefix of what was submitted; a kill costs at most the epoch
in flight.

The queue is bounded: if the filesystem cannot keep up, submit() blocks
rather than holding many epochs of weights in RAM. ``blocked_s`` records
the time spent blocked, logged per epoch in training_log.jsonl.
"""
from __future__ import annotations

import queue
import threading
import time


class AsyncCheckpointWriter:
    """FIFO background writer. submit() is called from one thread only.

    depth is counted in submissions, so submit one callable per epoch (see
    checkpoint.write_epoch_artifacts) rather than one per file.
    """

    def __init__(self, depth: int = 2, enabled: bool = True):
        self.enabled = enabled
        self.blocked_s = 0.0      # time submit() spent under backpressure
        self.last_write_s = 0.0   # duration of the most recent completed write
        self._error: BaseException | None = None
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if enabled:
            self._q = queue.Queue(maxsize=max(1, depth))
            self._thread = threading.Thread(target=self._run,
                                            name="ckpt-writer", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                fn, args, kwargs = item
                t0 = time.time()
                try:
                    fn(*args, **kwargs)
                except BaseException as e:      # surfaced by submit/drain/close
                    if self._error is None:
                        self._error = e
                else:
                    self.last_write_s = time.time() - t0
            finally:
                self._q.task_done()

    def _reraise(self) -> None:
        """Re-raise a write failure on the training thread, so a checkpoint
        that failed to land stops the run instead of surfacing at resume."""
        err, self._error = self._error, None
        if err is not None:
            raise err

    def submit(self, fn, *args, **kwargs) -> None:
        """Queue a write. Blocks (recording blocked_s) when the queue is full.

        Runs fn inline when disabled, so non-rank-0 callers and tests share
        one code path with no `if rank == 0` at the call site."""
        if not self.enabled:
            fn(*args, **kwargs)
            return
        self._reraise()
        t0 = time.time()
        self._q.put((fn, args, kwargs))
        self.blocked_s += time.time() - t0

    def drain(self) -> None:
        """Block until every queued write has landed. Call before anything
        reads these files back (the SWA average reads epoch_weights/)."""
        if not self.enabled:
            return
        self._q.join()
        self._reraise()

    def take_blocked_s(self) -> float:
        """Backpressure accumulated since the last call, for the epoch log."""
        v, self.blocked_s = self.blocked_s, 0.0
        return v

    def close(self) -> None:
        """Drain, stop the thread, re-raise any write failure. Idempotent."""
        if not self.enabled:
            return
        self._q.put(None)
        self._thread.join()
        self.enabled = False
        self._reraise()

    def __enter__(self) -> "AsyncCheckpointWriter":
        return self

    def __exit__(self, *exc) -> bool:
        # drain even when unwinding an exception: the completed epochs are
        # still valid resume points
        self.close()
        return False
