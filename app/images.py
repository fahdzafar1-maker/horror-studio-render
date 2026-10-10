"""Scene images with Gemini image models.

- Reference images (protagonist / entity / location) are made first and passed
  with every beat -> same face, same entity, same place across the episode.
- A file that already exists is never regenerated (re-runs cost nothing).
- Safety refusal -> one retry with the 'implied' fallback prompt (shadow,
  silhouette, aftermath). Still refused -> recorded as missing; the render reuses
  the previous image with a slower move.
"""
import base64
import io
import os
import shutil
import time

from PIL import Image, ImageDraw, ImageFilter

from . import gemini
from .config import MOCK_MODE
from .storage import job_dir, job_path, load_json, save_json


def img_path(job_id, image_id):
    return job_path(job_id, "images", f"{image_id}.png")


def _part(path):
    with open(path, "rb") as f:
        return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(f.read()).decode()}}


def _mock(prompt, image_id, aspect):
    w, h = (1920, 1080) if aspect == "16:9" else (1080, 1920) if aspect == "9:16" else (1024, 1024)
    seed = sum(map(ord, image_id))
    im = Image.new("RGB", (w, h), (8, 10, 14))
    dr = ImageDraw.Draw(im)
    for y in range(h):
        c = int(10 + 40 * (y / h))
        dr.line([(0, y), (w, y)], fill=(c // 2, c // 2 + seed % 20, c))
    cx, cy = (seed * 37) % w, h // 2
    dr.ellipse([cx - 120, cy - 300, cx + 120, cy + 300], fill=(5, 5, 5))
    im = im.filter(ImageFilter.GaussianBlur(3))
    dr = ImageDraw.Draw(im)
    dr.text((40, 40), f"{image_id}\n{prompt[:120]}", fill=(200, 200, 200))
    return im


def _call(model, prompt, refs, src, aspect, size):
    parts = []
    for r in refs:
        parts.append(_part(r))
    if src:
        parts.append(_part(src))
    parts.append({"text": prompt})
    gen = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}}
    if size:
        gen["imageConfig"]["imageSize"] = size
    resp = gemini.generate(model, {"contents": [{"role": "user", "parts": parts}],
                                   "generationConfig": gen}, timeout=240)
    for mime, data in gemini.inline_parts(resp):
        if mime.startswith("image/"):
            return Image.open(io.BytesIO(data)).convert("RGB"), ""
    return None, gemini.block_reason(resp) or "no image returned"


def generate(job_id: str, items: list, model: str, aspect: str = "16:9", size: str = "",
             throttle_s: float = 2.0, max_images: int = 80, progress=None) -> dict:
    """items: [{id, prompt, fallback_prompt?, refs?:[image_id], src?: image_id, model?, aspect?, size?}]"""
    job_dir(job_id, "images")
    log_path = job_path(job_id, "images", "_log.json")
    log = load_json(log_path, {}) or {}
    made = skipped = fell_back = reused = 0
    missing = []
    calls = 0
    for n, it in enumerate(items):
        iid = it["id"]
        out = img_path(job_id, iid)
        if os.path.exists(out):
            skipped += 1
            if progress:
                progress(n + 1, len(items), f"{iid} exists")
            continue
        # The story returns to a place already shown -> reuse that frame instead of paying for a new one.
        if it.get("copy_of") and os.path.exists(img_path(job_id, it["copy_of"])):
            shutil.copyfile(img_path(job_id, it["copy_of"]), out)
            reused += 1
            if progress:
                progress(n + 1, len(items), f"{iid} reuses {it['copy_of']}")
            continue
        if calls >= max_images:
            missing.append({"id": iid, "reason": "max_images cap reached"})
            continue
        a = it.get("aspect") or aspect
        m = it.get("model") or model
        sz = it.get("size", size)
        refs = [img_path(job_id, r) for r in it.get("refs", []) if os.path.exists(img_path(job_id, r))]
        src = img_path(job_id, it["src"]) if it.get("src") and os.path.exists(img_path(job_id, it["src"])) else None
        if MOCK_MODE:
            im, why = _mock(it["prompt"], iid, a), ""
        else:
            im, why = _call(m, it["prompt"], refs, src, a, sz)
            calls += 1
            if im is None and it.get("fallback_prompt"):
                time.sleep(throttle_s)
                im, why2 = _call(m, it["fallback_prompt"], refs, src, a, sz)
                calls += 1
                if im is not None:
                    fell_back += 1
                    log[iid] = {"fallback": True, "first_refusal": why}
                else:
                    why = f"{why} / fallback: {why2}"
            time.sleep(throttle_s)
        if im is None:
            missing.append({"id": iid, "reason": why})
        else:
            im.save(out, "PNG")
            made += 1
        save_json(log_path, log)
        if progress:
            progress(n + 1, len(items), f"{iid} {'ok' if im is not None else 'MISSING'}")
    return {"requested": len(items), "generated": made, "skipped": skipped, "reused": reused,
            "fallbacks": fell_back, "missing": missing, "paid_calls": calls}
