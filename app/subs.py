"""Subtitles from aligned word timings: SRT for YouTube, ASS for burned Shorts captions."""
from .config import font_family


def _ts(t, comma=True):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    sep = "," if comma else "."
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{ms:03d}"


def _ass_ts(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def cues_from_words(words, max_chars=42, max_lines=2, max_dur=5.0, offset=0.0):
    cues, cur, cur_sid = [], [], None
    limit = max_chars * max_lines

    def flush():
        if cur:
            cues.append({"start": cur[0]["start"] + offset, "end": cur[-1]["end"] + offset,
                         "text": " ".join(w["w"] for w in cur)})

    for w in words:
        text_len = len(" ".join(x["w"] for x in cur + [w]))
        new_sentence = cur_sid is not None and w["sid"] != cur_sid and cur and \
            cur[-1]["w"].rstrip('"”’\'').endswith((".", "?", "!"))
        too_long = text_len > limit or (cur and w["end"] - cur[0]["start"] > max_dur)
        if cur and (new_sentence or too_long):
            flush()
            cur = []
        cur.append(w)
        cur_sid = w["sid"]
    flush()
    # tidy: min gap, extend a little, wrap lines
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else c["end"] + 1
        c["end"] = min(max(c["end"] + 0.2, c["start"] + 0.8), nxt - 0.02)
        c["text"] = _wrap(c["text"], max_chars)
    return cues


def _wrap(text, max_chars):
    if len(text) <= max_chars:
        return text
    words = text.split()
    best, best_diff = text, 10 ** 9
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) <= max_chars + 4 and len(b) <= max_chars + 4:
            d = abs(len(a) - len(b))
            if d < best_diff:
                best, best_diff = a + "\n" + b, d
    return best


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{_ts(c['start'])} --> {_ts(c['end'])}\n{c['text']}\n\n")


def write_ass_kinetic(words, path, w=1080, h=1920, offset=0.0, chunk=3, margin_v=560, size=100):
    """Big 2-3 word captions for Shorts (current chunk only)."""
    fam = font_family("display")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fam},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,7,3,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    i = 0
    while i < len(words):
        grp = words[i:i + chunk]
        # don't carry a chunk across a sentence end
        for k, x in enumerate(grp):
            if x["w"].rstrip('"”’\'').endswith((".", "?", "!")) and k < len(grp) - 1:
                grp = grp[:k + 1]
                break
        s = grp[0]["start"] + offset
        e = (words[i + len(grp)]["start"] + offset) if i + len(grp) < len(words) else grp[-1]["end"] + offset + 0.3
        txt = " ".join(x["w"] for x in grp).upper().replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Cap,,0,0,0,,{{\\fad(60,0)}}{txt}")
        i += len(grp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
