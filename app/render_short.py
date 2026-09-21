"""Shorts (1080x1920): Part 1 Hook + Part 2 Threat are cut from the long-form mix;
Part 3 Twist+Subscribe uses its own voice block and an end card."""
import os
import random

from . import mixer, video
from .config import FPS, VW, VH, font, SR
from .images import img_path
from .storage import get_data, job_dir, job_path, put_data
from .subs import write_ass_kinetic
from .util import duration, ffmpeg


def _esc(t):
    return t.replace("\\", "").replace(":", " ").replace("'", "’").replace("%", " percent")


def _shots_for_window(job_id, beats, by_id, t0, t1, rnd, shots_dir, tag):
    """Vertical shots for [t0, t1] using the beats active in that window."""
    order = [b for b in beats if b.get("at") in by_id]
    order.sort(key=lambda b: by_id[b["at"]]["start"])
    marks = [(by_id[b["at"]]["start"], b) for b in order]
    active = [m for m in marks if m[0] <= t0]
    seq = ([active[-1]] if active else []) + [m for m in marks if t0 < m[0] < t1]
    if not seq:
        seq = [marks[0]]
    jobs, files = [], []
    last = None
    for i, (st, b) in enumerate(seq):
        s = max(st, t0)
        e = seq[i + 1][0] if i + 1 < len(seq) else t1
        p = img_path(job_id, b["id"])
        if os.path.exists(p):
            last = p
        if not last:
            continue
        for si, sd in enumerate(video.split_long(e - s, max_len=4.0)):
            f0, f1 = round((s - t0) * FPS), round((s - t0 + sd) * FPS)
            s += sd
            kind = "punch" if si % 2 else rnd.choice(["push_in", "pull_out"])
            mot = video.motion(kind, sd, 0.03, rnd)
            out = os.path.join(shots_dir, f"{tag}_{len(files):03d}.mp4")
            jobs.append(dict(img=last, out=out, frames=f1 - f0, mot=mot, w=VW, h=VH, vertical=True,
                             fade_in=0.25 if not files else 0, crf=20))
            files.append(out)
    video.render_many(jobs)
    return files


def _finish(silent, audio, ass, out, end_text=None, end_from=None, total=None):
    vf = [f"ass={ass}"]
    if end_text:
        big, small = end_text
        vf.append(f"drawtext=fontfile={font('display')}:text='{_esc(big)}':fontsize=110:fontcolor=white:"
                  f"borderw=6:bordercolor=black:x=(w-text_w)/2:y=h*0.40:enable='gte(t,{end_from:.2f})'")
        vf.append(f"drawtext=fontfile={font('display')}:text='{_esc(small)}':fontsize=58:fontcolor=white@0.9:"
                  f"borderw=4:bordercolor=black:x=(w-text_w)/2:y=h*0.40+140:enable='gte(t,{end_from:.2f})'")
    args = ["-i", silent, "-i", audio, "-map", "0:v", "-map", "1:a", "-vf", ",".join(vf),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-movflags", "+faststart"]
    if total:
        args += ["-t", f"{total:.3f}"]
    ffmpeg(*args, out)
    return out


def render(job_id: str, parts: list, channel_name: str = "", progress=None) -> dict:
    """parts: [{part:1|2, from, to}] and/or [{part:3, images:[beat ids]}]"""
    timings = get_data(job_id, "timings")
    beats = (get_data(job_id, "beats") or {}).get("visual_beats", [])
    by_id = {s["id"]: s for s in timings["sentences"]}
    rnd = random.Random(job_id + "shorts")
    work = job_dir(job_id, "shorts", "work")
    mix_final = job_path(job_id, "render", "mix_final.wav")
    results = {}
    for n, p in enumerate(parts):
        part = int(p["part"])
        out = job_path(job_id, "out", f"short_{part}.mp4")
        if part in (1, 2):
            a, b = by_id.get(p["from"]), by_id.get(p["to"])
            if not a or not b:
                raise RuntimeError(f"short {part}: sentence ids not found ({p.get('from')}..{p.get('to')})")
            t0 = max(0.0, a["start"] - 0.25)
            t1 = b["end"] + 0.9
            end_card = 2.4
            if t1 - t0 > 59.4 - end_card:  # keep the whole Short under 60 s: cut back to a sentence end
                ends = [s["end"] for s in timings["sentences"] if t0 < s["end"] <= t0 + 59.4 - end_card - 0.9]
                t1 = (max(ends) if ends else t0 + 56.0) + 0.9
            dur = t1 - t0 + end_card
            aud = os.path.join(work, f"s{part}.wav")
            ffmpeg("-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", mix_final, "-af",
                   f"afade=t=in:d=0.12,afade=t=out:st={t1 - t0 - 0.2:.2f}:d=0.8,apad",
                   "-t", f"{dur:.3f}", aud)
            files = _shots_for_window(job_id, beats, by_id, t0, t1 + end_card, rnd, work, f"p{part}")
            silent = video.concat(files, os.path.join(work, f"s{part}_v.mp4"))
            words = [dict(w, start=w["start"] - t0, end=w["end"] - t0) for w in timings["words"]
                     if w["start"] >= t0 - 0.05 and w["end"] <= t1 + 0.05]
            ass = os.path.join(work, f"s{part}.ass")
            write_ass_kinetic(words, ass)
            _finish(silent, aud, ass, out, ("FULL STORY", "on the channel now"), t1 - t0, dur)
        elif part == 3:
            t3 = get_data(job_id, "timings_short3")
            if not t3:
                raise RuntimeError("short 3 voice missing — run /tts/generate + /audio/finalize target=short3")
            voice = job_path(job_id, "audio", "short3", "voice.wav")
            first = t3["sentences"][0]["id"] if t3["sentences"] else None
            last = t3["sentences"][-1]["id"] if t3["sentences"] else None
            cues = [{"at": first, "cue": "drone_pulse", "volume": "mid"},
                    {"at": first, "cue": "sub_drop", "volume": "mid"}]
            if last:
                cues.append({"at": last, "cue": "riser_short", "volume": "low"})
            aud = os.path.join(work, "s3.wav")
            info = mixer.build_mix(voice, t3, cues, os.path.join(work, "s3_raw.wav"), aud,
                                   tail_silence=0.5, end_screen=4.5, auto_drone=False, seed=7)
            total = min(59.5, info["duration"])
            imgs = [i for i in p.get("images", []) if os.path.exists(img_path(job_id, i))]
            if not imgs:
                imgs = [b["id"] for b in beats if os.path.exists(img_path(job_id, b["id"]))][:4]
            seg = total / max(1, len(imgs))
            jobs, files = [], []
            for i, iid in enumerate(imgs):
                for si, sd in enumerate(video.split_long(seg, max_len=4.0)):
                    k = len(files)
                    f0 = round(sum(j["frames"] for j in jobs))
                    frames = round((i * seg + (si + 1) * sd) * FPS) - f0
                    kind = "punch" if si % 2 else "push_in"
                    outp = os.path.join(work, f"p3_{k:03d}.mp4")
                    jobs.append(dict(img=img_path(job_id, iid), out=outp, frames=frames,
                                     mot=video.motion(kind, sd, 0.03, rnd), w=VW, h=VH, vertical=True,
                                     fade_in=0.25 if k == 0 else 0, crf=20))
                    files.append(outp)
            video.render_many(jobs)
            silent = video.concat(files, os.path.join(work, "s3_v.mp4"))
            ass = os.path.join(work, "s3.ass")
            write_ass_kinetic(t3["words"], ass)
            _finish(silent, aud, ass, out, ("SUBSCRIBE", "full story on the channel"),
                    info["end_screen_start"], total)
        results[f"short_{part}"] = {"file": f"out/short_{part}.mp4", "duration": round(duration(out), 2)}
        if progress:
            progress(n + 1, len(parts), f"short {part} done")
    put_data(job_id, "render_shorts", results)
    return results
