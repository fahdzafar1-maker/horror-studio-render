"""Short moving clips (image -> video) with Google's Veo models on the same Gemini key.

A clip is made from a beat's own still, so the place and the entity match the rest of
the episode. render_long plays the clip at the start of that beat and falls back to the
still (Ken Burns) for the rest of the beat. A clip that already exists is never paid twice.
"""
import base64
import os
import time

import requests

from .config import GEMINI_API_KEY, GEMINI_BASE, MOCK_MODE
from .images import img_path
from .storage import job_path, load_json, save_json
from .util import duration, ffmpeg

API_ROOT = GEMINI_BASE.rsplit("/models", 1)[0]
NEG = ("bright colors, cartoon, anime, fast camera moves, distorted face, extra limbs, blurry, "
       "text, letters, watermark, blood, gore, wounds")


def clip_path(job_id, clip_id):
    return job_path(job_id, "clips", f"{clip_id}.mp4")


def _h():
    return {"x-goog-api-key": GEMINI_API_KEY}


def list_models() -> list:
    r = requests.get(f"{GEMINI_BASE}?pageSize=1000", headers=_h(), timeout=60)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", []) if "veo" in m["name"]]


def _find_uri(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "uri" and isinstance(v, str):
                return v
            u = _find_uri(v)
            if u:
                return u
    if isinstance(o, list):
        for v in o:
            u = _find_uri(v)
            if u:
                return u
    return ""


def _one(model, prompt, image_file, duration_s, timeout_s=900):
    with open(image_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {"instances": [{"prompt": prompt,
                           "image": {"bytesBase64Encoded": b64, "mimeType": "image/png"}}],
            "parameters": {"aspectRatio": "16:9", "negativePrompt": NEG,
                           "durationSeconds": int(duration_s), "personGeneration": "allow_adult"}}
    r = requests.post(f"{GEMINI_BASE}/{model}:predictLongRunning", json=body, headers=_h(), timeout=120)
    if r.status_code == 400 and "durationSeconds" in r.text:  # some models have a fixed length
        body["parameters"].pop("durationSeconds")
        r = requests.post(f"{GEMINI_BASE}/{model}:predictLongRunning", json=body, headers=_h(), timeout=120)
    if r.status_code != 200:
        return None, f"start HTTP {r.status_code}: {r.text[:300]}"
    op = r.json()["name"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(10)
        s = requests.get(f"{API_ROOT}/{op}", headers=_h(), timeout=60)
        if s.status_code != 200:
            continue
        j = s.json()
        if not j.get("done"):
            continue
        if j.get("error"):
            return None, f"error: {str(j['error'])[:300]}"
        uri = _find_uri(j.get("response", {}))
        if not uri:
            return None, f"no video (filtered?): {str(j.get('response'))[:300]}"
        v = requests.get(uri, headers=_h(), timeout=300, allow_redirects=True)
        if v.status_code != 200:
            return None, f"download HTTP {v.status_code}"
        return v.content, ""
    return None, "timed out"


def generate(job_id: str, items: list, model: str, duration_s: int = 8, max_clips: int = 20,
             progress=None) -> dict:
    """items: [{id (beat id), image_id?, prompt}] -> clips/<id>.mp4 (no audio)."""
    os.makedirs(os.path.dirname(clip_path(job_id, "x")), exist_ok=True)
    log_path = job_path(job_id, "clips", "_log.json")
    log = load_json(log_path, {}) or {}
    made, skipped, missing, calls = 0, 0, [], 0
    for n, it in enumerate(items):
        cid = it["id"]
        out = clip_path(job_id, cid)
        if os.path.exists(out):
            skipped += 1
            continue
        src = img_path(job_id, it.get("image_id") or cid)
        if not os.path.exists(src):
            missing.append({"id": cid, "reason": "source image missing"})
            continue
        if calls >= max_clips:
            missing.append({"id": cid, "reason": "max_clips cap reached"})
            continue
        if MOCK_MODE:
            ffmpeg("-loop", "1", "-i", src, "-t", str(duration_s), "-vf", "scale=1280:720", "-an", out)
            made += 1
            continue
        data, why = _one(it.get("model") or model, it["prompt"], src, duration_s)
        calls += 1
        if data is None:
            missing.append({"id": cid, "reason": why})
        else:
            tmp = out + ".raw.mp4"
            with open(tmp, "wb") as f:
                f.write(data)
            ffmpeg("-i", tmp, "-an", "-c:v", "copy", out)  # drop the model's own audio track
            os.remove(tmp)
            log[cid] = {"model": it.get("model") or model, "seconds": round(duration(out), 2)}
            save_json(log_path, log)
            made += 1
        if progress:
            progress(n + 1, len(items), f"{cid} {'ok' if data else 'MISSING'}")
    return {"requested": len(items), "generated": made, "skipped": skipped, "missing": missing,
            "paid_calls": calls}
