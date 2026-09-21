import json
import subprocess

import numpy as np
import soundfile as sf


def run(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}): {p.stderr[-1500:]}")
    return p


def ffmpeg(*args, timeout=3600):
    return run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], timeout=timeout)


def duration(path: str) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(p.stdout)["format"]["duration"])


def load_audio(path: str, sr: int, channels: int = 1) -> np.ndarray:
    """Decode any audio file to float32 (frames, channels) at sr using ffmpeg."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                        "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"],
                       capture_output=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"decode failed {path}: {p.stderr[-500:]!r}")
    a = np.frombuffer(p.stdout, dtype=np.float32)
    return a.reshape(-1, channels).copy()


def write_wav(path: str, audio: np.ndarray, sr: int, subtype="PCM_16"):
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype=subtype)


def db(x: float) -> float:
    return float(10 ** (x / 20.0))


def trim_silence(a: np.ndarray, sr: int, thresh_db=-45.0, keep=0.12):
    """Trim leading/trailing silence of a mono array, leaving `keep` seconds."""
    if a.size == 0:
        return a
    mono = np.abs(a if a.ndim == 1 else a.mean(axis=1))
    win = max(1, int(sr * 0.01))
    n = len(mono) // win
    if n == 0:
        return a
    env = mono[: n * win].reshape(n, win).max(axis=1)
    loud = np.where(env > db(thresh_db))[0]
    if loud.size == 0:
        return a
    s = max(0, loud[0] * win - int(keep * sr))
    e = min(len(a), (loud[-1] + 1) * win + int(keep * sr))
    return a[s:e]
