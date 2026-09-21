"""Tiny async task runner.

Every long operation (TTS, images, render) returns a task_id immediately; n8n polls
GET /tasks/{id} every 30 s. One heavy worker = renders never overlap = no OOM.
Task state is written to disk, so a restart turns 'running' into a clear error
instead of an endless 'running'.
"""
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .config import WORK_DIR
from .storage import save_json, load_json

_pool = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


def _dir():
    d = os.path.join(WORK_DIR, "tasks")
    os.makedirs(d, exist_ok=True)
    return d


def _path(tid):
    return os.path.join(_dir(), f"{tid}.json")


def get(tid: str):
    return load_json(_path(tid))


def _update(tid, **kw):
    with _lock:
        t = load_json(_path(tid), {}) or {}
        t.update(kw)
        t["updated_at"] = time.time()
        save_json(_path(tid), t)


class Progress:
    def __init__(self, tid):
        self.tid = tid

    def __call__(self, done, total, note=""):
        _update(self.tid, progress=f"{done}/{total}", note=note)


def submit(kind: str, job_id: str, fn, *args, **kwargs) -> dict:
    tid = f"{kind}-{uuid.uuid4().hex[:10]}"
    save_json(_path(tid), {"task_id": tid, "kind": kind, "job_id": job_id,
                           "status": "queued", "created_at": time.time()})

    def run():
        _update(tid, status="running", started_at=time.time())
        try:
            result = fn(*args, progress=Progress(tid), **kwargs)
            _update(tid, status="done", result=result, finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            _update(tid, status="error", error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-3000:], finished_at=time.time())

    _pool.submit(run)
    return {"task_id": tid, "status": "queued"}


def mark_interrupted():
    """Called on startup: anything left 'running'/'queued' died with the old container."""
    for f in os.listdir(_dir()):
        if not f.endswith(".json"):
            continue
        t = load_json(os.path.join(_dir(), f), {}) or {}
        if t.get("status") in ("running", "queued"):
            t["status"] = "error"
            t["error"] = "interrupted: service restarted. Re-run the stage (finished files are skipped)."
            save_json(os.path.join(_dir(), f), t)
