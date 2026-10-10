"""Background music library: whole tracks (e.g. from the YouTube Audio Library) played continuously
under the narration, Mr. Nightmare style. Files are loudness-normalised to -24 LUFS like the beds,
so the mixer's music level means the same thing for every track.

Files: music/<slug>.wav (44.1 kHz stereo) + music/_meta.json {slug: {title, artist, source}}.
"""
import json
import os
import random
import re

from .config import SR, WORK_DIR
from .storage import load_json, save_json
from .util import ffmpeg


def music_dir() -> str:
    d = os.path.join(WORK_DIR, "music")
    os.makedirs(d, exist_ok=True)
    return d


def _meta_path():
    return os.path.join(music_dir(), "_meta.json")


def listing() -> dict:
    meta = load_json(_meta_path(), {}) or {}
    have = sorted(f[:-4] for f in os.listdir(music_dir()) if f.endswith(".wav"))
    return {"tracks": [{"slug": s, **meta.get(s, {})} for s in have]}


def save_upload(filename: str, data: bytes, title: str = "", artist: str = "") -> dict:
    """filename like 'Black Mass - Brian Bolger.mp3' -> title/artist parsed from it when not given."""
    base = os.path.splitext(os.path.basename(filename))[0]
    if not title:
        title, _, a = base.partition(" - ")
        artist = artist or a
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")[:60] or "track"
    tmp = os.path.join(music_dir(), f"_upload_{slug}")
    with open(tmp, "wb") as f:
        f.write(data)
    out = os.path.join(music_dir(), f"{slug}.wav")
    ffmpeg("-i", tmp, "-af", "loudnorm=I=-24:TP=-3:LRA=11", "-ar", str(SR), "-ac", "2", out)
    os.remove(tmp)
    meta = load_json(_meta_path(), {}) or {}
    meta[slug] = {"title": title.strip(), "artist": artist.strip(), "source": "YouTube Audio Library"}
    save_json(_meta_path(), meta)
    return {"saved": slug, **meta[slug]}


def playlist(seed) -> list:
    """All tracks, shuffled per episode so consecutive videos start on different music."""
    files = [os.path.join(music_dir(), f) for f in sorted(os.listdir(music_dir())) if f.endswith(".wav")]
    random.Random(seed).shuffle(files)
    return files


def credits(paths) -> list:
    meta = load_json(_meta_path(), {}) or {}
    out = []
    for p in paths:
        m = meta.get(os.path.splitext(os.path.basename(p))[0], {})
        if m:
            out.append(f"{m.get('title', '')} — {m.get('artist', '')} ({m.get('source', '')})".strip())
    return out
