"""Per-job folders on the Railway volume. All heavy files live here, never inside n8n."""
import json
import os
import re
import shutil
import time

from .config import WORK_DIR, KEEP_DAYS

_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def safe(name: str) -> str:
    if not name or not _SAFE.match(name):
        raise ValueError(f"unsafe name: {name!r}")
    return name


def job_dir(job_id: str, *sub: str) -> str:
    d = os.path.join(WORK_DIR, "jobs", safe(job_id), *sub)
    os.makedirs(d, exist_ok=True)
    return d


def job_path(job_id: str, *parts: str) -> str:
    """Path to a file inside a job folder (parent folders created)."""
    p = os.path.join(WORK_DIR, "jobs", safe(job_id), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def data_path(job_id: str, name: str) -> str:
    return job_path(job_id, "data", safe(name) + ".json")


def get_data(job_id: str, name: str, default=None):
    return load_json(data_path(job_id, name), default)


def put_data(job_id: str, name: str, value) -> None:
    save_json(data_path(job_id, name), value)


def sfx_dir() -> str:
    d = os.path.join(WORK_DIR, "sfx")
    os.makedirs(d, exist_ok=True)
    return d


def cleanup_old_jobs() -> int:
    root = os.path.join(WORK_DIR, "jobs")
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - KEEP_DAYS * 86400
    n = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


def cleanup_intermediates(job_id: str) -> dict:
    """Delete big working files once the review package exists."""
    freed = 0
    base = os.path.join(WORK_DIR, "jobs", safe(job_id))
    for rel in ["render/shots", "render/video_noaudio.mp4", "render/mix_raw.wav",
                "audio/main/voice_raw.wav", "shorts/work"]:
        p = os.path.join(base, rel)
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                freed += sum(os.path.getsize(os.path.join(r, f)) for f in fs)
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
    return {"freed_mb": round(freed / 1e6, 1)}
