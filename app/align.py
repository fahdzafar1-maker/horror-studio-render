"""Sentence/word timing from the real voiceover.

Each TTS block is transcribed separately with faster-whisper (tiny.en, word
timestamps). The known script tokens of that block are aligned to the whisper
tokens with Needleman-Wunsch. Whisper's spelling mistakes don't matter — only its
clock is used. If whisper is unavailable, timing falls back to character share
inside the block (error stays inside one block, never compounds).
"""
import re
from difflib import SequenceMatcher

_model = None
_model_err = ""


def _whisper():
    global _model, _model_err
    if _model is None and not _model_err:
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        except Exception as e:  # noqa: BLE001
            _model_err = str(e)
    return _model


_norm_re = re.compile(r"[^a-z0-9']+")


def norm(w: str) -> str:
    w = w.lower().replace("’", "'").replace("‘", "'")
    return _norm_re.sub("", w).strip("'")


def tokenize(text: str):
    """Return list of (display_word, normalized) — em-dashes split words."""
    out = []
    for raw in re.split(r"\s+|—|–", text):
        raw = raw.strip()
        if not raw:
            continue
        n = norm(raw)
        if n:
            out.append((raw, n))
    return out


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def needleman(a, b):
    """Global alignment of token lists a (script) and b (asr). Returns list of (i, j|None)."""
    n, m = len(a), len(b)
    GAP = -1.0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up (gap in b), 2 left (gap in a)
    for i in range(1, n + 1):
        score[i][0] = i * GAP
        back[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * GAP
        back[0][j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            s = _sim(ai, b[j - 1])
            match = 2.0 if s == 1.0 else (1.0 if s >= 0.75 else -1.0)
            d = score[i - 1][j - 1] + match
            u = score[i - 1][j] + GAP
            l_ = score[i][j - 1] + GAP
            if d >= u and d >= l_:
                score[i][j], back[i][j] = d, 0
            elif u >= l_:
                score[i][j], back[i][j] = u, 1
            else:
                score[i][j], back[i][j] = l_, 2
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        bt = back[i][j]
        if i > 0 and j > 0 and bt == 0:
            ok = _sim(a[i - 1], b[j - 1]) >= 0.75
            pairs.append((i - 1, j - 1 if ok else None))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or bt == 1):
            pairs.append((i - 1, None))
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _charshare(tokens, dur):
    """Fallback: time proportional to characters."""
    total = sum(len(t[0]) + 1 for t in tokens) or 1
    t, out = 0.0, []
    for disp, _ in tokens:
        d = dur * (len(disp) + 1) / total
        out.append((t, t + d))
        t += d
    return out


def align_block(wav_path: str, sentences, dur: float):
    """sentences: [{id, text}] for this block. Returns (words, matched_ratio, source).

    words: [{w, start, end, sid}] with times relative to block start.
    """
    toks, sids = [], []
    for s in sentences:
        for tk in tokenize(s["text"]):
            toks.append(tk)
            sids.append(s["id"])
    if not toks:
        return [], 0.0, "empty"

    times = None
    ratio = 0.0
    source = "estimated"
    model = _whisper()
    if model is not None:
        try:
            segs, _ = model.transcribe(wav_path, word_timestamps=True, beam_size=1,
                                       vad_filter=False, language="en")
            asr = []
            for sg in segs:
                for w in (sg.words or []):
                    n = norm(w.word)
                    if n:
                        asr.append((n, float(w.start), float(w.end)))
            if asr:
                pairs = needleman([t[1] for t in toks], [a[0] for a in asr])
                times = [None] * len(toks)
                hit = 0
                for i, j in pairs:
                    if j is not None:
                        times[i] = (asr[j][1], asr[j][2])
                        hit += 1
                ratio = hit / len(toks)
                if ratio >= 0.5:
                    times = _interpolate(times, dur)
                    source = "measured"
                else:
                    times = None
        except Exception:  # noqa: BLE001
            times = None
    if times is None:
        times = _charshare(toks, dur)

    words = [{"w": toks[i][0], "start": round(times[i][0], 3),
              "end": round(times[i][1], 3), "sid": sids[i]} for i in range(len(toks))]
    return words, round(ratio, 3), source


def _interpolate(times, dur):
    n = len(times)
    known = [i for i, t in enumerate(times) if t is not None]
    out = list(times)
    for idx in range(n):
        if out[idx] is not None:
            continue
        prev = max([k for k in known if k < idx], default=None)
        nxt = min([k for k in known if k > idx], default=None)
        t0 = out[prev][1] if prev is not None else 0.0
        t1 = out[nxt][0] if nxt is not None else dur
        lo = prev if prev is not None else -1
        hi = nxt if nxt is not None else n
        span = hi - lo
        k = idx - lo
        a = t0 + (t1 - t0) * (k - 1) / max(1, span - 1)
        b = t0 + (t1 - t0) * k / max(1, span - 1)
        out[idx] = (a, max(a + 0.05, b))
    # force monotonic
    last = 0.0
    fixed = []
    for s, e in out:
        s = max(s, last)
        e = max(e, s + 0.04)
        fixed.append((s, min(e, dur)))
        last = s
    return fixed
