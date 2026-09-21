"""Horror sound library: built once (ElevenLabs sound generation) or uploaded by hand.

Files: sfx/<name>__<variant>.wav, 44.1 kHz stereo.
Beds are loudness-normalised to -24 LUFS, hits peak-normalised to -3 dBFS, so the
mixer's level settings mean the same thing for every file.
"""
import json
import os
import random
import time

import numpy as np
import requests

from .config import ELEVENLABS_API_KEY, MOCK_MODE, SR
from .storage import sfx_dir
from .util import ffmpeg, load_audio, write_wav, db

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "sfx_catalog.json")
SPECIAL = {"silence_cut"}


def catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        c = json.load(f)
    return {k: v for k, v in c.items() if not k.startswith("_")}


def variants(name: str):
    d = sfx_dir()
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith(name + "__") and f.endswith(".wav"))


def listing() -> dict:
    cat = catalog()
    have = {n: len(variants(n)) for n in cat}
    ready = sorted([n for n, c in have.items() if c > 0])
    return {"ready": ready + sorted(SPECIAL), "missing": sorted([n for n, c in have.items() if c == 0]),
            "counts": have, "types": {n: cat[n]["type"] for n in cat},
            "sfx_list_for_prompt": ", ".join(ready + sorted(SPECIAL))}


def _normalize(src: str, dst: str, kind: str):
    if kind == "bed":
        ffmpeg("-i", src, "-af", "loudnorm=I=-24:TP=-3:LRA=7", "-ar", str(SR), "-ac", "2", dst)
    else:
        a = load_audio(src, SR, 2)
        peak = float(np.abs(a).max() or 1.0)
        write_wav(dst, a * (db(-3) / peak), SR)


def _mock_sound(kind: str, dur: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal((n, 2)).astype(np.float32)
    # crude low-pass by cumulative average
    k = 200 if kind == "bed" else 40
    kernel = np.ones(k, np.float32) / k
    out = np.stack([np.convolve(noise[:, c], kernel, mode="same") for c in range(2)], axis=1)
    env = np.ones(n, np.float32) if kind == "bed" else np.exp(-np.linspace(0, 6, n)).astype(np.float32)
    return out * env[:, None] * 0.5


def build(names=None, n_bed: int = 2, n_hit: int = 3, force: bool = False,
          prompt_influence: float = 0.45, progress=None) -> dict:
    cat = catalog()
    todo = [n for n in (names or list(cat)) if n in cat]
    d = sfx_dir()
    made, skipped, errors = [], 0, []
    jobs = []
    for n in todo:
        k = n_bed if cat[n]["type"] == "bed" else n_hit
        for v in range(1, k + 1):
            jobs.append((n, v))
    for i, (n, v) in enumerate(jobs):
        out = os.path.join(d, f"{n}__{v}.wav")
        if os.path.exists(out) and not force:
            skipped += 1
            continue
        spec = cat[n]
        try:
            if MOCK_MODE:
                write_wav(out, _mock_sound(spec["type"], spec["duration"], hash((n, v)) % 9999), SR)
            else:
                if not ELEVENLABS_API_KEY:
                    raise RuntimeError("ELEVENLABS_API_KEY not set (or upload files with /sfx/upload)")
                body = {"text": spec["prompt"], "duration_seconds": spec["duration"],
                        "prompt_influence": prompt_influence}
                if spec["type"] == "bed":
                    body["loop"] = True
                r = None
                for attempt in range(4):
                    r = requests.post("https://api.elevenlabs.io/v1/sound-generation",
                                      params={"output_format": "mp3_44100_192"},
                                      headers={"xi-api-key": ELEVENLABS_API_KEY}, json=body, timeout=180)
                    if r.status_code == 200:
                        break
                    if r.status_code == 422 and "loop" in body:   # older model: no loop flag
                        body.pop("loop")
                        continue
                    if r.status_code in (429, 500, 502, 503):
                        time.sleep(15 * (attempt + 1))
                        continue
                    break
                if r.status_code != 200:
                    raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
                tmp = out + ".mp3"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                _normalize(tmp, out, spec["type"])
                os.remove(tmp)
                time.sleep(1.0)
            made.append(f"{n}__{v}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{n}__{v}: {e}")
        if progress:
            progress(i + 1, len(jobs), f"{n}__{v}")
    return {"made": len(made), "skipped": skipped, "errors": errors, **listing()}


def save_upload(name: str, variant: int, data: bytes) -> dict:
    cat = catalog()
    if name not in cat:
        raise ValueError(f"unknown sfx name {name}; add it to sfx_catalog.json first")
    d = sfx_dir()
    tmp = os.path.join(d, f"_upload_{name}")
    with open(tmp, "wb") as f:
        f.write(data)
    out = os.path.join(d, f"{name}__{variant}.wav")
    _normalize(tmp, out, cat[name]["type"])
    os.remove(tmp)
    return {"saved": os.path.basename(out)}


def pick(name: str, rnd: random.Random):
    v = variants(name)
    return rnd.choice(v) if v else None
