"""Voiceover: Gemini TTS per block (one phase style per block) + finalize/alignment."""
import base64
import os
import re
import time

import numpy as np
import soundfile as sf

from . import align, gemini
from .config import MOCK_MODE, SR
from .storage import job_dir, job_path, load_json, save_json, put_data
from .util import ffmpeg, trim_silence, write_wav

TTS_SR = 24000


def _block_path(job_id, target, i):
    return job_path(job_id, "audio", target, f"block_{i:03d}.wav")


def _mock_voice(text: str) -> np.ndarray:
    """Placeholder voice: soft tone bursts per word, ~14.5 chars/sec."""
    words = text.split()
    out = []
    rng = np.random.default_rng(len(text))
    for w in words:
        d = max(0.12, (len(w) + 1) / 14.5)
        t = np.arange(int(d * TTS_SR)) / TTS_SR
        f = 140 + rng.integers(0, 60)
        tone = 0.25 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
        out.append(tone.astype(np.float32))
        if w.endswith((".", "?", "!")):
            out.append(np.zeros(int(0.35 * TTS_SR), np.float32))
    return np.concatenate(out) if out else np.zeros(TTS_SR, np.float32)


def generate(job_id: str, target: str, blocks: list, model: str, voice: str,
             throttle_s: float = 6.0, reset: bool = False, progress=None) -> dict:
    """blocks: [{index, phase, style, sentences:[{id,text}]}]"""
    target = re.sub(r"[^a-z0-9_]", "", target) or "main"
    d = job_dir(job_id, "audio", target)
    if reset:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    save_json(os.path.join(d, "blocks.json"), blocks)
    made, skipped = 0, 0
    for n, b in enumerate(blocks):
        path = _block_path(job_id, target, b["index"])
        text = " ".join(s["text"].strip() for s in b["sentences"])
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            skipped += 1
            if progress:
                progress(n + 1, len(blocks), "skipped (exists)")
            continue
        if MOCK_MODE:
            pcm = _mock_voice(text)
            sf.write(path, pcm, TTS_SR)
        else:
            prompt = f"{b.get('style', '').strip()}\n\n{text}".strip()
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
                },
            }
            # Gemini TTS safety refusals on horror narration are stochastic: the same block
            # usually voices fine on a second or third try.
            parts, why = [], ""
            for attempt in range(4):
                resp = gemini.generate(model, body, timeout=300)
                parts = list(gemini.inline_parts(resp))
                if parts:
                    break
                why = gemini.block_reason(resp) or "no reason"
                if progress:
                    progress(n, len(blocks), f"block {b['index']} refused ({why}), retry {attempt + 1}")
                time.sleep(throttle_s + 3)
            if not parts:
                raise gemini.GeminiError(f"TTS returned no audio for block {b['index']} ({why}) after 4 tries")
            mime, raw = parts[0]
            rate = int((re.search(r"rate=(\d+)", mime) or [None, TTS_SR])[1])
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if rate != TTS_SR:
                tmp = path + ".raw.wav"
                sf.write(tmp, pcm, rate)
                ffmpeg("-i", tmp, "-ar", str(TTS_SR), "-ac", "1", path)
                os.remove(tmp)
            else:
                sf.write(path, pcm, TTS_SR)
            time.sleep(throttle_s)
        made += 1
        if progress:
            progress(n + 1, len(blocks), f"block {b['index']} phase {b.get('phase')}")
    return {"blocks": len(blocks), "generated": made, "skipped": skipped}


VOICE_FX = {
    # intimate close-mic narrator: remove rumble, gentle glue compression, a little presence & air
    "intimate": "highpass=f=75,lowpass=f=15500,"
                "acompressor=threshold=-21dB:ratio=2.6:attack=8:release=160:makeup=2,"
                "equalizer=f=180:t=q:w=1.2:g=1.5,equalizer=f=3200:t=q:w=1.4:g=1.8,"
                "equalizer=f=7500:t=q:w=2:g=-1.5",
    "clean": "highpass=f=70",
}


def finalize(job_id: str, target: str = "main", gaps: dict = None, voice_fx: str = "intimate",
             progress=None) -> dict:
    """Trim + join blocks with dramatic gaps, align words/sentences, write voice.wav (44.1k)."""
    gaps = gaps or {}
    g_default = float(gaps.get("default", 0.45))
    g_phase = float(gaps.get("phase_change", 1.1))
    after_phase = {str(k): float(v) for k, v in (gaps.get("after_phase") or {}).items()}

    d = job_dir(job_id, "audio", target)
    blocks = load_json(os.path.join(d, "blocks.json"))
    if not blocks:
        raise RuntimeError(f"no blocks.json for {job_id}/{target} — run /tts/generate first")

    pieces, words, sentences, block_info, phase_starts = [], [], [], [], {}
    t = 0.0
    sources, ratios = [], []
    for n, b in enumerate(blocks):
        path = _block_path(job_id, target, b["index"])
        if not os.path.exists(path):
            raise RuntimeError(f"missing audio block {b['index']}")
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        a = trim_silence(a, sr)
        tmp = path.replace(".wav", ".trim.wav")
        sf.write(tmp, a, sr)
        dur = len(a) / sr
        w, ratio, src = align.align_block(tmp, b["sentences"], dur)
        os.remove(tmp)
        sources.append(src)
        ratios.append(ratio)
        ph = str(b.get("phase", ""))
        if ph and ph not in phase_starts:
            phase_starts[ph] = round(t, 3)
        for x in w:
            words.append({**x, "start": round(x["start"] + t, 3), "end": round(x["end"] + t, 3)})
        block_info.append({"index": b["index"], "phase": b.get("phase"), "start": round(t, 3),
                           "end": round(t + dur, 3)})
        pieces.append(a)
        t += dur
        # gap after this block
        nxt = blocks[n + 1] if n + 1 < len(blocks) else None
        if nxt is not None:
            if str(nxt.get("phase")) != ph:
                gap = max(g_phase, after_phase.get(ph, 0.0))
            else:
                gap = g_default
            pieces.append(np.zeros(int(gap * sr), np.float32))
            t += gap
        if progress:
            progress(n + 1, len(blocks), f"aligned block {b['index']} ({src})")

    voice = np.concatenate(pieces)
    raw = job_path(job_id, "audio", target, "voice_raw.wav")
    sf.write(raw, voice, TTS_SR)
    out = job_path(job_id, "audio", target, "voice.wav")
    ffmpeg("-i", raw, "-af", VOICE_FX.get(voice_fx, VOICE_FX["clean"]) +
           ",loudnorm=I=-18:TP=-2:LRA=9", "-ar", str(SR), "-ac", "1", out)

    # sentence spans from words
    by_sid = {}
    for x in words:
        s = by_sid.setdefault(x["sid"], [x["start"], x["end"]])
        s[0] = min(s[0], x["start"])
        s[1] = max(s[1], x["end"])
    for b in blocks:
        for s in b["sentences"]:
            if s["id"] in by_sid:
                st, en = by_sid[s["id"]]
                sentences.append({"id": s["id"], "phase": b.get("phase"), "start": st, "end": en,
                                  "text": s["text"]})

    measured = sources.count("measured")
    timing_source = "measured" if measured == len(sources) else (
        "partial" if measured else "estimated")
    timings = {
        "target": target, "duration": round(t, 3), "timing_source": timing_source,
        "match_ratio": round(sum(ratios) / max(1, len(ratios)), 3),
        "sentences": sentences, "words": words, "blocks": block_info, "phase_starts": phase_starts,
        "whisper_error": align._model_err,
    }
    put_data(job_id, "timings" if target == "main" else f"timings_{target}", timings)
    return {"duration": timings["duration"], "timing_source": timing_source,
            "match_ratio": timings["match_ratio"], "sentences": len(sentences),
            "phase_starts": phase_starts, "whisper_error": align._model_err}


def voice_shorts(job_id: str, shorts: list, model: str, voice: str, progress=None) -> dict:
    """shorts: [{target, style, lines:[str]}] — one fast narration per Short, voiced + aligned in one task.

    Gemini TTS safety refusals are stochastic on horror text, so each Short gets 3 attempts
    before the task fails."""
    out = {}
    for n, s in enumerate(shorts):
        target = re.sub(r"[^a-z0-9_]", "", s["target"]) or f"short{n + 1}"
        blocks = [{"index": 0, "phase": target, "style": s.get("style", ""),
                   "sentences": [{"id": f"{target}_{i + 1:02d}", "text": t} for i, t in enumerate(s["lines"])]}]
        last = None
        for attempt in range(3):
            try:
                generate(job_id, target, blocks, model, voice, throttle_s=2, reset=True)
                last = None
                break
            except gemini.GeminiError as e:
                last = e
                time.sleep(5)
        if last:
            raise last
        out[target] = finalize(job_id, target, {"default": 0.25, "phase_change": 0.25}, "intimate")
        if progress:
            progress(n + 1, len(shorts), f"{target} voiced ({out[target]['duration']}s)")
    return out


def mock_b64_wav(text: str) -> str:  # used by tests only
    return base64.b64encode(_mock_voice(text).tobytes()).decode()
