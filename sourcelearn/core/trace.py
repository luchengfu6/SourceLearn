"""JSONL run tracing. Every LLM call, op execution, and coordination event is
recorded so that all efficiency/coordination metrics can be recomputed offline.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, log_dir: str | Path, experiment_id: str):
        self.dir = Path(log_dir) / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"
        self._fh = self.path.open("a")
        self._lock = threading.Lock()  # compile batches call LLMs in parallel

    def event(self, kind: str, **payload: Any) -> None:
        rec = {"ts": time.time(), "kind": kind, **payload}
        line = json.dumps(rec, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class NullTrace(Trace):
    """Trace that discards events."""

    def __init__(self):  # noqa: super().__init__ intentionally skipped
        pass

    def event(self, kind: str, **payload: Any) -> None:
        pass

    def close(self) -> None:
        pass
