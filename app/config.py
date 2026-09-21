"""Central settings. Everything comes from Railway environment variables."""
import os

WORK_DIR = os.environ.get("WORK_DIR", "/app/storage")
RENDER_KEY = os.environ.get("RENDER_KEY", "")            # shared secret with n8n (header x-render-key or ?key=)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")    # Tier-1 prepaid key (TTS + images)
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")  # only for the one-time SFX library build
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"      # 1 = no paid API calls (placeholder images/voice/sfx)
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", "30"))
RENDER_THREADS = int(os.environ.get("RENDER_THREADS", "0")) or max(1, (os.cpu_count() or 2) // 2)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

SR = 44100          # mix sample rate
FPS = 30
W, H = 1920, 1080   # long-form frame
VW, VH = 1080, 1920 # shorts frame

FONT_DIR = os.environ.get("FONT_DIR", "/app/fonts")


def font(name: str) -> str:
    """Return a font path; falls back to DejaVu if the downloaded font is missing."""
    candidates = {
        "display": [f"{FONT_DIR}/Anton-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "sub": [f"{FONT_DIR}/Oswald-SemiBold.ttf", f"{FONT_DIR}/Oswald-Variable.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    }[name]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


def font_family(name: str) -> str:
    p = font(name)
    if "Anton" in p:
        return "Anton"
    if "Oswald" in p:
        return "Oswald"
    return "DejaVu Sans"
