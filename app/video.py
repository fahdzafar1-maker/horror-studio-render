"""Shot building shared by long-form and Shorts: Ken Burns moves + horror grade."""
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .config import FPS, RENDER_THREADS, WORK_DIR
from .util import ffmpeg

FX_DIR = os.path.join(WORK_DIR, "fx")
FX_OPACITY = {"fog": 0.30, "rain": 0.38}


def fx_asset(kind: str) -> str:
    """Free atmosphere layers, made once and cached: a drifting fog plate (wide PNG) and a looping
    rain clip. They are screen-blended over a shot, so black = no change."""
    os.makedirs(FX_DIR, exist_ok=True)
    if kind == "fog":
        out = os.path.join(FX_DIR, "fog_v1.png")
        if not os.path.exists(out):
            rng = np.random.default_rng(7)
            small = (rng.random((27, 96)) ** 2.2 * 255).astype(np.uint8)
            im = Image.fromarray(small, "L").resize((3840, 1080), Image.BICUBIC)
            from PIL import ImageFilter
            im = im.filter(ImageFilter.GaussianBlur(40))
            a = np.asarray(im).astype(np.float32)
            a = (a - a.min()) / max(1.0, a.max() - a.min()) * 150
            # the plate wraps around: blend its two ends so the drift has no seam
            ramp = np.linspace(0, 1, 400)[None, :]
            a[:, -400:] = a[:, -400:] * (1 - ramp) + a[:, :400] * ramp
            Image.fromarray(a.astype(np.uint8), "L").convert("RGB").save(out)
        return out
    if kind == "rain":
        out = os.path.join(FX_DIR, "rain_v1.mp4")
        if not os.path.exists(out):
            from PIL import ImageDraw
            tmp = os.path.join(FX_DIR, "rain_frames")
            os.makedirs(tmp, exist_ok=True)
            rng = np.random.default_rng(11)
            n = 30  # one-second loop: every drop travels a whole multiple of the 1180 px wrap
            drops = [(rng.uniform(0, 2100), rng.uniform(0, 1180), 1180 / n * int(rng.integers(1, 3)),
                      rng.uniform(30, 75), int(rng.uniform(70, 170))) for _ in range(420)]
            for f in range(n):
                im = Image.new("RGB", (1920, 1080), (0, 0, 0))
                d = ImageDraw.Draw(im)
                for x, y, speed, ln, c in drops:
                    yy = (y + speed * f) % 1180 - 100
                    xx = x - yy * 0.18
                    d.line([(xx, yy), (xx - ln * 0.18, yy + ln)], fill=(c, c, c), width=1)
                im.save(os.path.join(tmp, f"r{f:03d}.png"))
            ffmpeg("-framerate", str(FPS), "-i", os.path.join(tmp, "r%03d.png"), "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "18", out)
        return out
    raise ValueError(f"unknown fx {kind}")


def bars_crop(img):
    """Image models sometimes paint cinematic letterbox bars into a 16:9 still.
    Return an ffmpeg crop that removes them ('' if there are none).

    A bar = leading rows that are near-black, ending in a SHARP step up to the picture
    (a dark night sky fades in gradually and is left alone). Letterbox bars come in
    matching pairs, so a single dark band (e.g. a dashboard) is never cropped."""
    a = np.asarray(Image.open(img).convert("L")).astype(np.float32)
    H, W = a.shape

    def bar(rows):
        n = 0
        while n < H // 4 and rows[n].mean() < 12 and np.percentile(rows[n], 90) < 22:
            n += 1
        if n < 0.02 * H or n + 6 >= H:
            return 0
        return n if rows[n + 6].mean() - rows[max(0, n - 3)].mean() > 8 else 0

    top, bottom = bar(a), bar(a[::-1])
    if not top or not bottom or abs(top - bottom) > 0.25 * max(top, bottom):
        return ""
    return f"crop={W}:{H - top - bottom}:0:{top},"


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
    debar = bars_crop(img)
    if vertical:
        pre = f"scale=-2:{H2},crop={W2}:{H2}:(iw-{W2})/2:0"
    else:
        pre = f"scale={W2}:{H2}:force_original_aspect_ratio=increase,crop={W2}:{H2}"
    tmp = out + ".tmp.png"
    ffmpeg("-i", img, "-vf", debar + pre + ",eq=contrast=1.08:brightness=-0.025:saturation=0.78,"
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


def shake(amp):
    """Handheld unease: overscan 4 % and drift the crop a few pixels (two incommensurate sines)."""
    return (f"scale=iw*1.04:-2,crop=iw/1.04:ih/1.04:"
            f"x='(in_w-out_w)/2+{amp:.1f}*sin(n*0.83)+{amp * 0.5:.1f}*sin(n*2.3)':"
            f"y='(in_h-out_h)/2+{amp * 0.7:.1f}*cos(n*0.61)+{amp * 0.4:.1f}*sin(n*1.9)'")


def render_shot(img, out, frames, mot, w, h, fade_in=0.0, fade_out=0.0, flicker=False,
                grain=0, crf=21, vertical=False, shake_amp=0.0, fx=None):
    z0, z1, x0, x1, y0, y1 = mot
    N = max(1, frames)
    still = prepare_still(img, w, h, vertical)
    zp = (f"zoompan=z='{z0:.4f}+({z1:.4f}-{z0:.4f})*on/{N}':"
          f"x='(iw-iw/zoom)*({x0:.3f}+({x1:.3f}-{x0:.3f})*on/{N})':"
          f"y='(ih-ih/zoom)*({y0:.3f}+({y1:.3f}-{y0:.3f})*on/{N})':"
          f"d={N}:s={w}x{h}:fps={FPS}")
    vf = [zp] + ([shake(shake_amp), f"scale={w}:{h}"] if shake_amp > 0 else []) + \
        ([grade(flicker, grain)] if (flicker or grain) else [])
    dur = N / FPS
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.2f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(0, dur - fade_out):.2f}:d={fade_out:.2f}")
    vf.append("format=yuv420p")
    enc = ["-frames:v", str(N), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-maxrate", "5M", "-bufsize", "10M", "-g", str(FPS * 2), "-an", out]
    if fx in FX_OPACITY and not vertical:
        base = [x for x in vf if not x.startswith("fade=") and x != "format=yuv420p"]
        post = [x for x in vf if x.startswith("fade=")] + ["format=yuv420p"]
        if fx == "fog":
            src = ["-loop", "1", "-framerate", str(FPS), "-i", fx_asset("fog")]
            layer = f"crop={w}:{h}:x='mod(t*22\\,3840-{w})':y=0"
        else:
            src = ["-stream_loop", "-1", "-i", fx_asset("rain")]
            layer = f"scale={w}:{h}"
        fc = (f"[0:v]{','.join(base)},format=gbrp[b];[1:v]{layer},fps={FPS},format=gbrp[f];"
              f"[b][f]blend=all_mode=screen:all_opacity={FX_OPACITY[fx]}:shortest=1,{','.join(post)}[v]")
        ffmpeg("-i", still, *src, "-filter_complex", fc, "-map", "[v]", *enc, timeout=1800)
        return out
    ffmpeg("-i", still, "-vf", ",".join(vf), *enc, timeout=1800)
    return out


def render_clip(clip, out, frames, w, h, fade_in=0.0, fade_out=0.0, grain=0, crf=21, vertical=False):
    """A moving clip as one shot: same frame size, grade and vignette as the stills, no audio.
    A clip shorter than the shot holds its last frame."""
    N = max(1, frames)
    dur = N / FPS
    if vertical:
        fit = f"scale=-2:{h},crop={w}:{h}"
    else:
        fit = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    vf = [f"fps={FPS}", fit, "eq=contrast=1.08:brightness=-0.025:saturation=0.78",
          "colorbalance=rs=-0.02:bs=0.035:rm=-0.01:bm=0.02", "vignette=PI/4.5",
          f"tpad=stop_mode=clone:stop_duration={dur:.2f}", "setsar=1"]
    if grain:
        vf.append(f"noise=alls={int(grain)}:allf=t")
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.2f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(0, dur - fade_out):.2f}:d={fade_out:.2f}")
    vf.append("format=yuv420p")
    ffmpeg("-i", clip, "-vf", ",".join(vf), "-frames:v", str(N), "-r", str(FPS),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-maxrate", "5M", "-bufsize", "10M", "-g", str(FPS * 2), "-an", out, timeout=1800)
    return out


def render_many(jobs):
    """jobs: list of kwargs for render_shot (or render_clip when the job has 'clip').
    Runs in parallel threads (ffmpeg does the work)."""
    seen = set()
    for k in jobs:  # grade each still once, serially (no two threads writing one file)
        if k.get("clip"):
            continue
        key = (k["img"], k["w"], k["h"], k.get("vertical", False))
        if key not in seen:
            seen.add(key)
            prepare_still(*key)

    def one(k):
        return render_clip(**k) if k.get("clip") else render_shot(**k)
    with ThreadPoolExecutor(max_workers=RENDER_THREADS) as ex:
        list(ex.map(one, jobs))


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
