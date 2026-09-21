"""Shot building shared by long-form and Shorts: Ken Burns moves + horror grade."""
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .config import FPS, RENDER_THREADS
from .util import ffmpeg


def prepare_still(img, w, h, vertical=False):
    """Crop to 2x frame size + bake the horror grade and vignette into the still ONCE.

    Per-frame colorbalance/vignette filters were 5x slower than the zoom itself;
    baking them into the still keeps rendering ~2x faster than real time per thread.
    """
    W2, H2 = w * 2, h * 2
    tag = f"{os.path.splitext(os.path.basename(img))[0]}_{W2}x{H2}.png"
    out = os.path.join(os.path.dirname(img), "graded", tag)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(img):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if vertical:
        pre = f"scale=-2:{H2},crop={W2}:{H2}:(iw-{W2})/2:0"
    else:
        pre = f"scale={W2}:{H2}:force_original_aspect_ratio=increase,crop={W2}:{H2}"
    tmp = out + ".tmp.png"
    ffmpeg("-i", img, "-vf", pre + ",eq=contrast=1.08:brightness=-0.025:saturation=0.78,"
           "colorbalance=rs=-0.02:bs=0.035:rm=-0.01:bm=0.02", "-frames:v", "1", tmp)
    im = np.asarray(Image.open(tmp).convert("RGB")).astype(np.float32)
    yy, xx = np.mgrid[0:H2, 0:W2]
    d = np.sqrt(((xx - W2 / 2) / (W2 / 2)) ** 2 + ((yy - H2 / 2) / (H2 / 2)) ** 2)
    shade = 1 - np.clip((d - 0.45) * 0.75, 0, 0.78) ** 1.2
    Image.fromarray((im * shade[..., None]).clip(0, 255).astype(np.uint8)).save(out)
    os.remove(tmp)
    return out


def grade(flicker=False, grain=0):
    g = []
    if flicker:
        g.append("eq=eval=frame:gamma='1+0.04*(random(0)-0.5)'")
    if grain:
        g.append(f"noise=alls={int(grain)}:allf=t")
    return ",".join(g)


def motion(kind: str, dur: float, speed: float, rnd: random.Random):
    """Return (z0, z1, x0, x1, y0, y1); x/y are fractions of the free space."""
    dz = min(0.28, max(0.05, speed * dur))
    if kind == "push_in":
        return 1.0, 1 + dz, 0.5, 0.5 + rnd.uniform(-.08, .08), 0.5, 0.45
    if kind == "pull_out":
        return 1 + dz, 1.0, 0.5 + rnd.uniform(-.1, .1), 0.5, 0.45, 0.5
    if kind == "pan_lr":
        return 1.14, 1.14 + dz / 3, 0.2, 0.8, 0.5, 0.5
    if kind == "pan_rl":
        return 1.14, 1.14 + dz / 3, 0.8, 0.2, 0.5, 0.5
    if kind == "drift_up":
        return 1.12, 1.12 + dz / 2, 0.5, 0.5, 0.75, 0.3
    # punch: tighter crop on a region, slow push
    x = rnd.uniform(0.25, 0.75)
    y = rnd.uniform(0.3, 0.6)
    return 1.3, 1.3 + dz, x, x + rnd.uniform(-.06, .06), y, y


def pick_motion(shot_hint: str, idx: int, sub: int, rnd: random.Random):
    hint = (shot_hint or "").lower()
    if sub > 0:
        return "punch"
    if "extreme" in hint or "close" in hint:
        return rnd.choice(["push_in", "punch", "push_in"])
    if "wide" in hint:
        return rnd.choice(["pan_lr", "pan_rl", "pull_out", "drift_up"])
    if "pov" in hint:
        return rnd.choice(["push_in", "drift_up"])
    return ["push_in", "pan_lr", "pull_out", "push_in", "pan_rl", "drift_up"][idx % 6]


def render_shot(img, out, frames, mot, w, h, fade_in=0.0, fade_out=0.0, flicker=False,
                grain=0, crf=21, vertical=False):
    z0, z1, x0, x1, y0, y1 = mot
    N = max(1, frames)
    still = prepare_still(img, w, h, vertical)
    zp = (f"zoompan=z='{z0:.4f}+({z1:.4f}-{z0:.4f})*on/{N}':"
          f"x='(iw-iw/zoom)*({x0:.3f}+({x1:.3f}-{x0:.3f})*on/{N})':"
          f"y='(ih-ih/zoom)*({y0:.3f}+({y1:.3f}-{y0:.3f})*on/{N})':"
          f"d={N}:s={w}x{h}:fps={FPS}")
    vf = [zp] + ([grade(flicker, grain)] if (flicker or grain) else [])
    dur = N / FPS
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.2f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(0, dur - fade_out):.2f}:d={fade_out:.2f}")
    vf.append("format=yuv420p")
    ffmpeg("-i", still, "-vf", ",".join(vf), "-frames:v", str(N), "-r", str(FPS),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-maxrate", "5M", "-bufsize", "10M", "-g", str(FPS * 2), "-an", out, timeout=1800)
    return out


def render_many(jobs):
    """jobs: list of kwargs for render_shot. Runs in parallel threads (ffmpeg does the work)."""
    seen = set()
    for k in jobs:  # grade each still once, serially (no two threads writing one file)
        key = (k["img"], k["w"], k["h"], k.get("vertical", False))
        if key not in seen:
            seen.add(key)
            prepare_still(*key)
    with ThreadPoolExecutor(max_workers=RENDER_THREADS) as ex:
        list(ex.map(lambda k: render_shot(**k), jobs))


def concat(files, out):
    lst = out + ".txt"
    with open(lst, "w") as f:
        for p in files:
            f.write(f"file '{os.path.abspath(p)}'\n")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out)
    os.remove(lst)
    return out


def split_long(dur, max_len=9.0):
    if dur <= max_len:
        return [dur]
    k = math.ceil(dur / 8.0)
    return [dur / k] * k
