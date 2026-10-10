"""Horror sound design mixer (sample-accurate, numpy).

Layers
  1. Voice (processed, -18 LUFS)                                  centre
  2. Beds  (ambience chosen by the script: room tone, wind, car...)  loop until next bed cue
  3. Hits  (one-shots placed in the pause BEFORE their sentence)      light stereo pan
  4. Auto dread drone: rises from Pressure (phase 6) to end of Climax (phase 9)
  5. 'Vacuum': all ambience drops to silence for 1.6 s when the Climax breaks
  6. silence_cut cue: ambience vanishes for 4 s (silence is the scariest sound)
  7. Ending: ambience fades after the last word, >= 2.5 s of true silence, then outro sting
Beds duck ~7 dB under the voice. Final master: two-pass loudnorm to -14 LUFS / -1 dBTP.
"""
import json
import random
import re

import numpy as np

from . import sfx
from .config import SR
from .util import db, ffmpeg, load_audio, run, write_wav

CR = 100  # control rate (Hz) for envelopes

DEFAULT_LEVELS = {
    # beds/drone sit ~3 dB further under the narration than before (measured voice-over-background
    # was 18-19 dB median, 13 dB worst case on episode 4; target 20+ dB for a clear voice)
    "bed_low": -10.0, "bed_mid": -7.0,     # on -24 LUFS files
    "hit_low": -13.0, "hit_mid": -8.0,     # on -3 dBFS-peak files
    "drone_start": -18.0, "drone_end": -10.0,
    "duck_beds": -10.0, "duck_hits": -3.0,
    "end_screen_bed": -12.0,
}
PAN_CUES = {"footsteps_gravel", "footsteps_wood", "knock_three", "scratch_wall", "glass_tap",
            "tree_branch_scrape", "door_creak", "distant_dog", "whisper_indistinct", "breath_close"}


def _ramp(arr, t0, t1, v0, v1):
    i0, i1 = int(t0 * CR), int(t1 * CR)
    i0, i1 = max(0, i0), min(len(arr), i1)
    if i1 > i0:
        arr[i0:i1] = np.linspace(v0, v1, i1 - i0)


def _apply_gain_curve(ctrl, start_s, n):
    """Upsample a control-rate curve to samples for [start_s, start_s + n/SR)."""
    t = start_s + np.arange(n) / SR
    return np.interp(t * CR, np.arange(len(ctrl)), ctrl).astype(np.float32)


def _loop_to(a, n, xfade_s=1.0):
    """Tile a stereo bed to n samples with equal-power crossfades."""
    if len(a) >= n:
        return a[:n].copy()
    xf = min(int(xfade_s * SR), len(a) // 3)
    out = np.zeros((n, a.shape[1]), np.float32)
    pos = 0
    fin = np.sqrt(np.linspace(0, 1, xf))[:, None].astype(np.float32)
    fout = np.sqrt(np.linspace(1, 0, xf))[:, None].astype(np.float32)
    first = True
    while pos < n:
        seg = a.copy()
        if not first:
            seg[:xf] *= fin
        if len(seg) > xf:
            seg[-xf:] *= fout
        m = min(len(seg), n - pos)
        out[pos:pos + m] += seg[:m]
        pos += len(seg) - xf
        first = False
    return out


def _pan(a, p):
    """Equal-power pan of a stereo (n,2) array, p in [-1, 1]."""
    ang = (p + 1) * np.pi / 4
    return a * np.array([np.cos(ang) * 1.414, np.sin(ang) * 1.414], np.float32)


def build_mix(voice_path: str, timings: dict, cues: list, out_raw: str, out_final: str,
              tail_silence=2.8, end_screen=20.0, auto_drone=True, levels=None, seed=1,
              voice_offset=0.0) -> dict:
    L = {**DEFAULT_LEVELS, **(levels or {})}
    rnd = random.Random(seed)
    voice = load_audio(voice_path, SR, 1)[:, 0]
    v_end = voice_offset + len(voice) / SR
    total = v_end + tail_silence + end_screen
    N = int(total * SR) + 1
    mix = np.zeros((N, 2), np.float32)
    vs = int(voice_offset * SR)
    mix[vs:vs + len(voice)] += voice[:, None]

    # ---- control curves
    nC = int(total * CR) + 2
    hop = SR // CR
    vv = np.abs(voice)
    nh = len(vv) // hop
    rms = np.sqrt((vv[: nh * hop].reshape(nh, hop) ** 2).mean(axis=1)) if nh else np.zeros(0)
    active = np.zeros(nC, np.float32)
    off = int(voice_offset * CR)
    active[off:off + nh] = (rms > db(-42)).astype(np.float32)
    # attack 30 ms / release 450 ms smoothing
    sm = np.zeros_like(active)
    a_att, a_rel = 1 - np.exp(-1 / (0.03 * CR)), 1 - np.exp(-1 / (0.45 * CR))
    s = 0.0
    for i, x in enumerate(active):
        s += (x - s) * (a_att if x > s else a_rel)
        sm[i] = s
    duck_bed = 1 - sm * (1 - db(L["duck_beds"]))
    duck_hit = 1 - sm * (1 - db(L["duck_hits"]))

    bed_mask = np.ones(nC, np.float32)
    # ending: fade ambience after last word, keep silence
    _ramp(bed_mask, v_end, v_end + 1.5, 1, 0)
    bed_mask[int((v_end + 1.5) * CR):] = 0

    sent = {s["id"]: s for s in timings["sentences"]}
    order = [s["id"] for s in timings["sentences"]]

    def t_of(sid):
        s = sent.get(sid)
        return None if s is None else voice_offset + s["start"]

    def gap_before(sid):
        i = order.index(sid) if sid in order else -1
        if i <= 0:
            return voice_offset
        return voice_offset + sent[order[i - 1]]["end"]

    # silence_cut cues
    placed, skipped = [], []
    for c in cues:
        if c.get("cue") == "silence_cut":
            t = t_of(c.get("at"))
            if t is None:
                skipped.append(c)
                continue
            _ramp(bed_mask, t - 0.3, t, 1, 0)
            bed_mask[int(t * CR):int((t + 4.0) * CR)] = 0
            _ramp(bed_mask, t + 4.0, t + 6.0, 0, 1)
            placed.append({"cue": "silence_cut", "t": round(t, 2)})

    # climax vacuum + auto drone
    ps = {str(k): v + voice_offset for k, v in (timings.get("phase_starts") or {}).items()}
    t_pressure, t_reveal = ps.get("6"), ps.get("10")
    if t_reveal:
        _ramp(bed_mask, t_reveal - 0.9, t_reveal - 0.75, 1, 0)
        bed_mask[int((t_reveal - 0.75) * CR):int((t_reveal + 0.85) * CR)] = 0
        _ramp(bed_mask, t_reveal + 0.85, t_reveal + 3.0, 0, 1)

    # ---- beds
    types = sfx.catalog()
    bed_cues = []
    for c in cues:
        name = c.get("cue")
        if types.get(name, {}).get("type") == "bed":
            t = t_of(c.get("at"))
            if t is None:
                skipped.append(c)
                continue
            bed_cues.append((t, name, c.get("volume", "low")))
    bed_cues.sort()
    if not bed_cues or bed_cues[0][0] > 4.0:
        bed_cues.insert(0, (0.0, "room_tone_night", "low"))
    for i, (t, name, vol) in enumerate(bed_cues):
        f = sfx.pick(name, rnd)
        if not f:
            skipped.append({"cue": name, "why": "no file"})
            continue
        t0 = max(0.0, t - 0.75)
        t1 = (bed_cues[i + 1][0] + 0.75) if i + 1 < len(bed_cues) else v_end + 1.5
        n = int((t1 - t0) * SR)
        if n <= 0:
            continue
        a = _loop_to(load_audio(f, SR, 2), n)
        env = np.ones(n, np.float32)
        fi = min(n // 2, int((0.3 if t0 == 0 else 1.5) * SR))
        env[:fi] = np.linspace(0, 1, fi)
        fo = min(n // 2, int(1.5 * SR))
        env[-fo:] = np.linspace(1, 0, fo)
        g = db(L["bed_mid"] if vol == "mid" else L["bed_low"])
        s0 = int(t0 * SR)
        curve = _apply_gain_curve(duck_bed * bed_mask, t0, n) * env * g
        mix[s0:s0 + n] += a * curve[:, None]
        placed.append({"cue": name, "t": round(t, 2), "until": round(t1, 2)})

    if auto_drone and t_pressure and t_reveal and t_reveal > t_pressure + 5:
        f = sfx.pick("drone_dread", rnd) or sfx.pick("drone_pulse", rnd)
        if f:
            t0, t1 = t_pressure, t_reveal - 0.75
            n = int((t1 - t0) * SR)
            a = _loop_to(load_audio(f, SR, 2), n)
            rise = np.linspace(db(L["drone_start"]), db(L["drone_end"]), n).astype(np.float32)
            fi = min(n // 3, int(4 * SR))
            rise[:fi] *= np.linspace(0, 1, fi)
            rise[-int(0.12 * SR):] *= np.linspace(1, 0, int(0.12 * SR))
            curve = _apply_gain_curve(1 - sm * (1 - db(-4)), t0, n) * rise
            s0 = int(t0 * SR)
            mix[s0:s0 + n] += a * curve[:, None]
            placed.append({"cue": "auto_drone", "t": round(t0, 2), "until": round(t1, 2)})

    # ---- hits
    for c in cues:
        name = c.get("cue")
        if types.get(name, {}).get("type") != "hit":
            continue
        st = t_of(c.get("at"))
        if st is None:
            skipped.append(c)
            continue
        f = sfx.pick(name, rnd)
        if not f:
            skipped.append({"cue": name, "why": "no file"})
            continue
        t = max(gap_before(c["at"]) + 0.05, st - 0.35)
        a = load_audio(f, SR, 2)
        if name in PAN_CUES:
            a = _pan(a, rnd.uniform(-0.35, 0.35))
        s0 = int(t * SR)
        n = min(len(a), N - s0)
        if n <= 0:
            continue
        g = db(L["hit_mid"] if c.get("volume") == "mid" else L["hit_low"])
        curve = _apply_gain_curve(duck_hit, t, n) * g
        mix[s0:s0 + n] += a[:n] * curve[:, None]
        placed.append({"cue": name, "t": round(t, 2)})

    # ---- ending: outro sting + quiet end-screen bed
    t_end = v_end + tail_silence
    f = sfx.pick("outro_sting", rnd)
    if f and end_screen > 0:
        a = load_audio(f, SR, 2)
        s0 = int(t_end * SR)
        n = min(len(a), N - s0)
        mix[s0:s0 + n] += a[:n] * db(-8)
    f = sfx.pick("room_tone_night", rnd)
    if f and end_screen > 2:
        n = int(end_screen * SR)
        a = _loop_to(load_audio(f, SR, 2), n)
        env = np.ones(n, np.float32)
        env[: 2 * SR] = np.linspace(0, 1, 2 * SR)
        env[-3 * SR:] = np.linspace(1, 0, 3 * SR)
        s0 = int(t_end * SR)
        n = min(n, N - s0)
        mix[s0:s0 + n] += a[:n] * (env[:n] * db(L["end_screen_bed"]))[:, None]

    write_wav(out_raw, mix, SR, subtype="FLOAT")
    loud = master(out_raw, out_final)
    return {"duration": round(total, 3), "voice_end": round(v_end, 3), "end_screen_start": round(t_end, 3),
            "cues_placed": len(placed), "cues_skipped": skipped[:30], "placements": placed, "loudness": loud}


def master(src: str, dst: str, target_i=-14.0, target_tp=-1.0) -> dict:
    """Two-pass EBU R128 loudness normalisation (linear, no pumping)."""
    p = run(["ffmpeg", "-hide_banner", "-i", src, "-af",
             f"loudnorm=I={target_i}:TP={target_tp}:LRA=11:print_format=json", "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    meas = json.loads(m.group(0)) if m else {}
    if meas:
        af = (f"loudnorm=I={target_i}:TP={target_tp}:LRA=11:measured_I={meas['input_i']}:"
              f"measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:"
              f"measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}:linear=true")
    else:
        af = f"loudnorm=I={target_i}:TP={target_tp}:LRA=11"
    ffmpeg("-i", src, "-af", af, "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", dst)
    return {"input_i": meas.get("input_i"), "input_tp": meas.get("input_tp"), "target_i": target_i}
