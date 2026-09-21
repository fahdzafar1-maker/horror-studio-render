"""One mobile-friendly page per episode: watch, copy, download, tick the checklist."""
import html
import os

from .config import PUBLIC_BASE_URL, RENDER_KEY
from .storage import get_data, job_path


def _u(job_id, rel):
    return f"{PUBLIC_BASE_URL}/files/{job_id}/{rel}?key={RENDER_KEY}"


def _box(label, text, rows=3):
    t = html.escape(text or "")
    return (f"<div class=box><div class=lab>{html.escape(label)} "
            f"<button onclick=\"navigator.clipboard.writeText(this.parentNode.nextElementSibling.value)\">copy</button>"
            f"</div><textarea rows={rows} readonly>{t}</textarea></div>")


def page(job_id: str) -> str:
    seo = get_data(job_id, "seo", {}) or {}
    rl = get_data(job_id, "render_long", {}) or {}
    thumbs = get_data(job_id, "thumbs", {}) or {}
    shorts = get_data(job_id, "render_shorts", {}) or {}
    qa = rl.get("qa", {})
    parts = [f"<h1>{html.escape(seo.get('title_main', job_id))}</h1>",
             f"<p class=meta>{html.escape(job_id)} · {rl.get('duration', 0)//60:.0f} min · "
             f"sync {'OK' if qa.get('sync_ok') else 'CHECK'} · {qa.get('size_mb', '?')} MB · "
             f"loudness in {rl.get('mix', {}).get('loudness', {}).get('input_i', '?')} → -14 LUFS</p>"]
    if os.path.exists(job_path(job_id, "out", "master.mp4")):
        parts.append(f"<video controls preload=metadata src='{_u(job_id, 'out/master.mp4')}'></video>"
                     f"<p><a href='{_u(job_id, 'out/master.mp4')}' download>Download master.mp4</a> · "
                     f"<a href='{_u(job_id, 'out/subtitles.srt')}' download>subtitles.srt</a></p>")
    tm = [("main", thumbs.get("main")), ("alt", thumbs.get("alt"))]
    parts.append("<h2>Thumbnails (A/B)</h2><div class=row>")
    for lab, t in tm:
        if t:
            parts.append(f"<figure><img src='{_u(job_id, t)}'><figcaption>{lab} · "
                         f"<a href='{_u(job_id, t)}' download>download</a></figcaption></figure>")
    parts.append("</div>")
    parts.append("<h2>Upload text</h2>")
    parts.append(_box("Title (main)", seo.get("title_main"), 1))
    parts.append(_box("Title (alt — Test & Compare)", seo.get("title_alt"), 1))
    parts.append(_box("Description", seo.get("description"), 14))
    parts.append(_box("Tags", ", ".join(seo.get("tags", [])), 3))
    parts.append("<h2>Shorts</h2><div class=row>")
    for k in ("short_1", "short_2", "short_3"):
        s = shorts.get(k)
        if s:
            meta = (seo.get("shorts") or {}).get(k, {})
            parts.append(f"<figure><video controls preload=metadata src='{_u(job_id, s['file'])}'></video>"
                         f"<figcaption>{k} · {s['duration']}s · <a href='{_u(job_id, s['file'])}' download>download</a>"
                         f"</figcaption>{_box('Short title', meta.get('title'), 1)}"
                         f"{_box('Short description', meta.get('description'), 3)}</figure>")
    parts.append("</div><h2>Publish checklist</h2><ul class=check>")
    for item in seo.get("checklist", []):
        parts.append(f"<li><label><input type=checkbox> {html.escape(item)}</label></li>")
    parts.append("</ul>")
    if seo.get("problems"):
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in seo["problems"]) + "</ul>")
    css = """body{background:#0d0f12;color:#e6e6e6;font:16px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:980px}
h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:26px 0 8px;color:#ff6b6b}.meta{color:#9aa}
video,img{width:100%;border-radius:8px;background:#000}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
figure{margin:0}figcaption{font-size:13px;color:#aaa;margin:4px 0}a{color:#7fc4ff}
.box{margin:8px 0}.lab{font-size:13px;color:#aaa;display:flex;justify-content:space-between}
textarea{width:100%;box-sizing:border-box;background:#161a20;color:#eee;border:1px solid #2a2f38;border-radius:6px;padding:8px;font:14px/1.4 monospace}
button{background:#2a2f38;color:#eee;border:0;border-radius:4px;padding:2px 10px}.check li{list-style:none;margin:6px 0}"""
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Review {html.escape(job_id)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
