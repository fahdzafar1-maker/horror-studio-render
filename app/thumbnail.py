"""Thumbnail compositor (1280x720). Source = the episode's own scene art (never stock).

Text rules are enforced by n8n before calling this (0-3 words, never repeats title).
"""
import base64
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import font
from .images import img_path
from .storage import job_path, put_data

TW, TH = 1280, 720


def _cover(im, w, h, focus_x=0.5, focus_y=0.45):
    r = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * r + 1), int(im.height * r + 1)), Image.LANCZOS)
    x = int((im.width - w) * focus_x)
    y = int((im.height - h) * focus_y)
    return im.crop((x, y, x + w, y + h))


def _vignette(im, strength=0.65):
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1 - strength * np.clip(d - 0.35, 0, None) ** 1.4, 0.15, 1)
    arr = np.asarray(im).astype(np.float32) * mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def compose(job_id: str, items: list) -> dict:
    """items: [{id, image_id, text, side: left|right, color: white|yellow|red}]"""
    out = []
    for it in items:
        src = img_path(job_id, it["image_id"])
        if not os.path.exists(src):
            out.append({"id": it["id"], "error": f"missing image {it['image_id']}"})
            continue
        im = Image.open(src).convert("RGB")
        side = it.get("side", "left")
        text = (it.get("text") or "").strip().upper()
        # subject away from the text
        im = _cover(im, TW, TH, focus_x=0.62 if side == "left" and text else 0.5)
        im = ImageEnhance.Contrast(im).enhance(1.28)
        im = ImageEnhance.Color(im).enhance(0.82)
        im = ImageEnhance.Sharpness(im).enhance(1.4)
        im = _vignette(im)
        if text:
            # darken behind text for readability at 168x94
            grad = Image.new("L", (TW, TH), 0)
            gd = ImageDraw.Draw(grad)
            for x in range(TW // 2):
                a = int(170 * (1 - x / (TW / 2)))
                xx = x if side == "left" else TW - 1 - x
                gd.line([(xx, 0), (xx, TH)], fill=a)
            im = Image.composite(Image.new("RGB", (TW, TH), (0, 0, 0)), im, grad)
            words = text.split()[:3]
            fcol = {"yellow": (255, 214, 10), "red": (230, 30, 30)}.get(it.get("color"), (245, 245, 245))
            dr = ImageDraw.Draw(im)
            max_w, max_h = int(TW * 0.56), int(TH * 0.78)
            # try 1 line, then 1 word per line; shrink font until it fits the text half
            best = None
            for lines in ([" ".join(words)], words):
                for size in range(210, 70, -6):
                    f = ImageFont.truetype(font("display"), size)
                    ws = [dr.textbbox((0, 0), ln, font=f, stroke_width=10)[2] for ln in lines]
                    hs = [dr.textbbox((0, 0), ln, font=f, stroke_width=10)[3] for ln in lines]
                    if max(ws) <= max_w and sum(hs) + 10 * (len(lines) - 1) <= max_h:
                        if best is None or size > best[1]:
                            best = (lines, size, f, ws, hs)
                        break
            lines, size, f, ws, hs = best or ([" ".join(words)], 72, ImageFont.truetype(font("display"), 72), [max_w], [80])
            y = (TH - sum(hs) - 10 * (len(lines) - 1)) // 2
            for ln, wdt, hgt in zip(lines, ws, hs):
                x = 55 if side == "left" else TW - wdt - 55
                dr.text((x + 6, y + 8), ln, font=f, fill=(0, 0, 0), stroke_width=10, stroke_fill=(0, 0, 0))
                dr.text((x, y), ln, font=f, fill=fcol, stroke_width=10, stroke_fill=(0, 0, 0))
                y += hgt + 10
        path = job_path(job_id, "out", f"thumb_{it['id']}.jpg")
        im.save(path, "JPEG", quality=92)
        small = im.resize((640, 360), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=85)
        out.append({"id": it["id"], "file": f"out/thumb_{it['id']}.jpg",
                    "size_kb": os.path.getsize(path) // 1024,
                    "preview_b64": base64.b64encode(buf.getvalue()).decode()})
    put_data(job_id, "thumbs_composed", [{k: v for k, v in o.items() if k != "preview_b64"} for o in out])
    return {"thumbs": out}
