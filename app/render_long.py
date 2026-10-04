"""Long-form render: shots from visual beats (real timings) + horror mix + SRT + chapters."""
import os
import random

from . import mixer, video
from .config import FPS, W, H, font, SR
from .images import img_path
from .storage import get_data, job_dir, job_path, put_data
from .subs import cues_from_words, write_srt
from .util import duration, ffmpeg


def _beat_order(beats, sentence_index):
    vb = [b for b in beats if b.get("at") in sentence_index]
    vb.sort(key=lambda b: sentence_index[b["at"]])
    return vb


def render(job_id: str, options: dict = None, progress=None) -> dict:
    o = {"tail_silence": 2.8, "end_screen": 20.0, "burn_subtitles": False, "crf": 21, "grain": 4,
         "auto_drone": True, "levels": {}, "channel_name": "", **(options or {})}
    timings = get_data(job_id, "timings")
    beats_doc = get_data(job_id, "beats")
    if not timings or not beats_doc:
        raise RuntimeError("timings/beats missing — run W4 (voice) and W2 (beats) first")
    sents = timings["sentences"]
    sidx = {s["id"]: i for i, s in enumerate(sents)}
    by_id = {s["id"]: s for s in sents}
    beats = _beat_order(beats_doc.get("visual_beats", []), sidx)
    if not beats:
        raise RuntimeError("no visual beats match the timed sentences")
    rnd = random.Random(job_id)
    wd = job_dir(job_id, "render")
    shots_dir = job_dir(job_id, "render", "shots")

    # ---------- audio first (defines total length)
    if progress:
        progress(0, 4, "mixing audio")
    voice = job_path(job_id, "audio", "main", "voice.wav")
    mix_final = os.path.join(wd, "mix_final.wav")
    mix_info = mixer.build_mix(voice, timings, beats_doc.get("sound_cues", []),
                               os.path.join(wd, "mix_raw.wav"), mix_final,
                               tail_silence=o["tail_silence"], end_screen=o["end_screen"],
                               auto_drone=o["auto_drone"], levels=o["levels"], seed=hash(job_id) % 10**6)
    total = mix_info["duration"]
    v_end = mix_info["voice_end"]
    end_start = mix_info["end_screen_start"]

    # ---------- shots
    phase_of = {s["id"]: int(s.get("phase") or 0) for s in sents}
    starts = [0.0] + [by_id[b["at"]]["start"] for b in beats[1:]]
    ends = starts[1:] + [end_start]
    jobs, files, missing = [], [], []
    last_img = None
    # first available image for leading gaps
    avail = [b for b in beats if os.path.exists(img_path(job_id, b["id"]))]
    if not avail:
        raise RuntimeError("no beat images found — run W5 first")
    last_img = img_path(job_id, avail[0]["id"])
    k = 0
    prev_phase = None
    for i, b in enumerate(beats):
        p = img_path(job_id, b["id"])
        if os.path.exists(p):
            last_img = p
        else:
            missing.append(b["id"])
        ph = phase_of.get(b["at"], 0)
        dur = ends[i] - starts[i]
        if dur <= 0.05:
            continue
        subs = video.split_long(dur)
        speed = 0.012 if ph >= 6 else 0.007
        t = starts[i]
        for si, sd in enumerate(subs):
            f0, f1 = round(t * FPS), round((t + sd) * FPS)
            t += sd
            kind = video.pick_motion(b.get("shot", ""), i, si, rnd)
            mot = video.motion(kind, sd, speed, rnd)
            out = os.path.join(shots_dir, f"s{k:04d}.mp4")
            is_last = (i == len(beats) - 1 and si == len(subs) - 1)
            jobs.append(dict(img=last_img, out=out, frames=f1 - f0, mot=mot, w=W, h=H,
                             fade_in=(0.4 if k == 0 else (0.6 if (si == 0 and ph != prev_phase) else 0)),
                             fade_out=(1.6 if is_last else 0),
                             # dread builds visually: flicker on entity frames from Pressure on,
                             # on every third frame of the Point-blank/Climax; handheld shake at the climax
                             flicker=(ph >= 6 and b.get("subject") == "entity") or (ph in (8, 9) and k % 3 == 0),
                             shake_amp=(3.0 if ph in (8, 9) else 0.0),
                             grain=o["grain"], crf=o["crf"]))
            files.append(out)
            k += 1
        prev_phase = ph

    # end screen: blurred last image + channel name (space for YouTube end-screen elements)
    es = os.path.join(shots_dir, "endscreen.mp4")
    n_es = round(total * FPS) - round(end_start * FPS)
    name = (o.get("channel_name") or "").replace(":", " ").replace("'", "").replace("%", "")
    dt = (f",drawtext=fontfile={font('display')}:text='{name}':fontsize=64:fontcolor=white@0.85:"
          f"x=(w-text_w)/2:y=90") if name else ""
    if progress:
        progress(1, 4, f"rendering {len(jobs)} shots")
    video.render_many(jobs)
    if n_es > 0:
        ffmpeg("-loop", "1", "-i", last_img, "-vf",
               f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2,"
               f"eq=brightness=-0.22:saturation=0.5{dt},fade=t=in:st=0:d=1.2,format=yuv420p",
               "-frames:v", str(n_es), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(o["crf"]), "-g", str(FPS * 2), "-an", es)
        files.append(es)

    if progress:
        progress(2, 4, "concat + subtitles")
    silent = video.concat(files, os.path.join(wd, "video_noaudio.mp4"))

    # ---------- subtitles
    cues = cues_from_words(timings["words"])
    srt = job_path(job_id, "out", "subtitles.srt")
    write_srt(cues, srt)

    # ---------- mux
    if progress:
        progress(3, 4, "mux")
    master = job_path(job_id, "out", "master.mp4")
    if o["burn_subtitles"]:
        style = (f"FontName=DejaVu Sans,FontSize=15,PrimaryColour=&H00F0F0F0,OutlineColour=&H00000000,"
                 f"BorderStyle=1,Outline=1.6,Shadow=0,MarginV=38")
        ffmpeg("-i", silent, "-i", mix_final, "-map", "0:v", "-map", "1:a",
               "-vf", f"subtitles={srt}:force_style='{style}'", "-c:v", "libx264", "-preset", "medium",
               "-crf", str(o["crf"]), "-maxrate", "5M", "-bufsize", "10M", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-movflags", "+faststart", master)
    else:
        ffmpeg("-i", silent, "-i", mix_final, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
               "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-movflags", "+faststart", master)

    # ---------- chapters (phase starts) + QA
    ps = timings.get("phase_starts", {})
    real = duration(master)
    qa = {"expected_s": round(total, 2), "actual_s": round(real, 2),
          "sync_ok": abs(real - total) < 0.6, "size_mb": round(os.path.getsize(master) / 1e6, 1),
          "missing_images": missing, "shots": len(jobs),
          "duration_ok": 11 * 60 <= real <= 19 * 60}
    result = {"master": "out/master.mp4", "srt": "out/subtitles.srt", "duration": round(real, 2),
              "voice_end": v_end, "end_screen_start": end_start, "phase_starts": ps,
              "mix": {k: v for k, v in mix_info.items() if k != "placements"}, "qa": qa}
    put_data(job_id, "render_long", {**result, "placements": mix_info["placements"]})
    if progress:
        progress(4, 4, "done")
    return result
