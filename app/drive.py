"""Archive finished episode files to Google Drive without routing them through n8n.

n8n (which holds the Google OAuth credential) opens one *resumable upload session* per
file and sends us the session URLs. A session URL needs no auth header, so we just
stream the file to it. Heavy files never touch n8n's memory.
"""
import os

import requests

from .storage import get_data, job_path


def write_upload_text(job_id: str) -> str:
    """out/upload_text.txt — everything needed to publish, in one file next to the video."""
    seo = get_data(job_id, "seo", {}) or {}
    shorts = seo.get("shorts") or {}
    lines = [f"TITLE (main): {seo.get('title_main', '')}", f"TITLE (alt, Test & Compare): {seo.get('title_alt', '')}",
             "", "DESCRIPTION:", seo.get("description", ""), "", "TAGS:", ", ".join(seo.get("tags", [])), ""]
    for k in sorted(shorts):
        lines += [f"{k.upper()} title: {shorts[k].get('title', '')}", f"{k.upper()} description: {shorts[k].get('description', '')}", ""]
    if seo.get("summary_ur"):
        lines += ["KAHANI KA KHULASA (Roman Urdu, sirf aap ke liye):", seo["summary_ur"], ""]
    if seo.get("checklist"):
        lines += ["PUBLISH CHECKLIST:"] + [f"[ ] {c}" for c in seo["checklist"]]
    p = job_path(job_id, "out", "upload_text.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "out/upload_text.txt"


def upload(job_id: str, files: list, progress=None) -> dict:
    """files: [{rel, upload_url}] — rel is a path inside the job folder (e.g. out/master.mp4)."""
    done, skipped = [], []
    for n, f in enumerate(files):
        rel = f["rel"]
        if ".." in rel:
            raise ValueError(f"bad path {rel}")
        if rel == "out/upload_text.txt":
            write_upload_text(job_id)
        p = job_path(job_id, *rel.split("/"))
        if not os.path.isfile(p):
            skipped.append(rel)
            continue
        with open(p, "rb") as fh:
            r = requests.put(f["upload_url"], data=fh, timeout=3600,
                             headers={"Content-Length": str(os.path.getsize(p))})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Drive upload of {rel} failed: HTTP {r.status_code} {r.text[:300]}")
        done.append({"rel": rel, "drive_id": r.json().get("id"), "mb": round(os.path.getsize(p) / 1e6, 1)})
        if progress:
            progress(n + 1, len(files), f"uploaded {rel}")
    return {"uploaded": done, "skipped": skipped}
