"""Horror Studio render service (FastAPI).

n8n = brain (prompts, checks, sheet state). This service = hands (voice, images,
sound, video). Heavy files never pass through n8n.
"""
import os
import shutil

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from . import drive, gemini, images, render_long, render_short, review, sfx, tasks, thumbnail, tts
from .config import MOCK_MODE, RENDER_KEY, WORK_DIR, PUBLIC_BASE_URL, font
from .storage import cleanup_intermediates, cleanup_old_jobs, get_data, job_path, put_data, safe

app = FastAPI(title="Horror Studio Render", version="1.0")


@app.on_event("startup")
def _startup():
    os.makedirs(WORK_DIR, exist_ok=True)
    tasks.mark_interrupted()
    cleanup_old_jobs()


def _auth(req: Request):
    if not RENDER_KEY:
        return
    k = req.headers.get("x-render-key") or req.query_params.get("key")
    if k != RENDER_KEY:
        raise HTTPException(401, "bad or missing render key")


# ------------------------------------------------------------------ health
@app.get("/health")
def health():
    free = shutil.disk_usage(WORK_DIR).free / 1e9 if os.path.exists(WORK_DIR) else None
    return {"ok": True, "mock_mode": MOCK_MODE, "work_dir": WORK_DIR, "disk_free_gb": round(free or 0, 1),
            "public_base_url": PUBLIC_BASE_URL, "whisper": _whisper_ok(),
            "fonts": {"display": font("display"), "sub": font("sub")}}


def _whisper_ok():
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ job data (JSON docs)
@app.put("/jobs/{job_id}/data/{name}")
def put_doc(job_id: str, name: str, request: Request, value=Body(...)):
    _auth(request)
    put_data(job_id, name, value)
    return {"saved": name}


@app.get("/jobs/{job_id}/data/{name}")
def get_doc(job_id: str, name: str, request: Request):
    _auth(request)
    return get_data(job_id, name, {}) or {}


@app.get("/jobs/{job_id}/bundle")
def bundle(job_id: str, names: str, request: Request):
    _auth(request)
    return {n: (get_data(job_id, safe(n), None)) for n in names.split(",") if n}


@app.post("/jobs/{job_id}/cleanup")
def cleanup(job_id: str, request: Request):
    _auth(request)
    return cleanup_intermediates(job_id)


# ------------------------------------------------------------------ tasks
@app.get("/tasks/{task_id}")
def task(task_id: str, request: Request):
    _auth(request)
    t = tasks.get(safe(task_id))
    if not t:
        raise HTTPException(404, "unknown task")
    return t


# ------------------------------------------------------------------ text LLM proxy
@app.post("/llm")
def llm(request: Request, p: dict = Body(...)):
    """n8n sends {model, body}; we add the key and handle 429 retryDelay / 503 retries."""
    _auth(request)
    try:
        return gemini.generate(p["model"], p["body"], timeout=int(p.get("timeout", 600)),
                               max_attempts=int(p.get("max_attempts", 5)))
    except gemini.GeminiError as e:
        raise HTTPException(502, f"Gemini: {e}")


# ------------------------------------------------------------------ voice
@app.post("/tts/generate")
def tts_generate(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("tts", p["job_id"], tts.generate, p["job_id"], p.get("target", "main"),
                        p["blocks"], p["model"], p["voice"], float(p.get("throttle_s", 6)),
                        bool(p.get("reset", False)))


@app.post("/shorts/voice")
def shorts_voice(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("svoice", p["job_id"], tts.voice_shorts, p["job_id"], p["shorts"], p["model"], p["voice"])


@app.post("/audio/finalize")
def audio_finalize(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("final", p["job_id"], tts.finalize, p["job_id"], p.get("target", "main"),
                        p.get("gaps"), p.get("voice_fx", "intimate"))


# ------------------------------------------------------------------ images
@app.post("/images/generate")
def images_generate(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("img", p["job_id"], images.generate, p["job_id"], p["items"], p["model"],
                        p.get("aspect", "16:9"), p.get("size", ""), float(p.get("throttle_s", 2)),
                        int(p.get("max_images", 80)))


@app.get("/jobs/{job_id}/images")
def list_images(job_id: str, request: Request):
    _auth(request)
    d = os.path.dirname(job_path(job_id, "images", "x"))
    return {"images": sorted(f[:-4] for f in os.listdir(d) if f.endswith(".png"))}


# ------------------------------------------------------------------ sound library
@app.get("/sfx/list")
def sfx_list(request: Request):
    _auth(request)
    return sfx.listing()


@app.post("/sfx/build")
def sfx_build(request: Request, p: dict = Body(default={})):
    _auth(request)
    return tasks.submit("sfx", "library", sfx.build, p.get("names"), int(p.get("variants_bed", 2)),
                        int(p.get("variants_hit", 3)), bool(p.get("force", False)),
                        float(p.get("prompt_influence", 0.45)))


@app.post("/sfx/upload/{name}")
async def sfx_upload(name: str, request: Request, variant: int = 1):
    _auth(request)
    return sfx.save_upload(safe(name), variant, await request.body())


# ------------------------------------------------------------------ render
@app.post("/render/long")
def render_long_ep(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("long", p["job_id"], render_long.render, p["job_id"], p.get("options", {}))


@app.post("/render/shorts")
def render_shorts_ep(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("shorts", p["job_id"], render_short.render, p["job_id"], p["parts"],
                        p.get("channel_name", ""))


@app.post("/thumbnail")
def thumb(request: Request, p: dict = Body(...)):
    _auth(request)
    return thumbnail.compose(p["job_id"], p["items"])


# ------------------------------------------------------------------ Google Drive archive
@app.get("/jobs/{job_id}/sizes")
def sizes(job_id: str, names: str, request: Request):
    """Byte sizes of finished files, so n8n can open Drive upload sessions of the right length."""
    _auth(request)
    out = {}
    for rel in (n for n in names.split(",") if n):
        if ".." in rel:
            raise HTTPException(400, "bad path")
        if rel == "out/upload_text.txt":
            drive.write_upload_text(safe(job_id))
        p = job_path(safe(job_id), *rel.split("/"))
        out[rel] = os.path.getsize(p) if os.path.isfile(p) else None
    return out


@app.post("/drive/upload")
def drive_upload(request: Request, p: dict = Body(...)):
    _auth(request)
    return tasks.submit("drive", p["job_id"], drive.upload, safe(p["job_id"]), p["files"])


# ------------------------------------------------------------------ review + files
@app.get("/review/{job_id}", response_class=HTMLResponse)
def review_page(job_id: str, request: Request):
    _auth(request)
    return review.page(safe(job_id))


@app.get("/files/{job_id}/{rel:path}")
def files(job_id: str, rel: str, request: Request):
    _auth(request)
    if ".." in rel:
        raise HTTPException(400, "bad path")
    p = job_path(safe(job_id), *rel.split("/"))
    if not os.path.isfile(p):
        raise HTTPException(404, "not found")
    return FileResponse(p)
