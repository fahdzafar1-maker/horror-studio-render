"""Gemini REST calls with honest 429 handling.

Lesson from Hidden America: n8n's retry caps at 5 s, a 429 wants ~40 s.
Here we read Google's own retryDelay and wait exactly that long.
"""
import base64
import re
import time

import requests

from .config import GEMINI_API_KEY, GEMINI_BASE


class GeminiError(Exception):
    pass


class GeminiBlocked(GeminiError):
    """Safety refusal (no retry with the same prompt)."""


def _retry_delay(resp) -> float:
    try:
        for d in resp.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(re.sub(r"[^0-9.]", "", d["retryDelay"]) or 30)
    except Exception:  # noqa: BLE001
        pass
    return 30.0


def generate(model: str, body: dict, timeout: int = 300, max_attempts: int = 6) -> dict:
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not set on the render service")
    url = f"{GEMINI_BASE}/{model}:generateContent"
    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, json=body, timeout=timeout,
                              headers={"x-goog-api-key": GEMINI_API_KEY})
        except requests.RequestException as e:
            last = str(e)
            time.sleep(min(60, 5 * attempt))
            continue
        if r.status_code == 200:
            return r.json()
        last = f"HTTP {r.status_code}: {r.text[:400]}"
        if r.status_code == 429:
            if "free_tier" in r.text:
                raise GeminiError("429 on a FREE-TIER key. Put the Tier-1 prepaid key in GEMINI_API_KEY. " + last)
            time.sleep(_retry_delay(r) + 2)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(min(90, 10 * attempt))
            continue
        raise GeminiError(last)  # 400/403/404: retry will not help
    raise GeminiError(f"gave up after {max_attempts} attempts: {last}")


def inline_parts(resp: dict):
    """Yield (mime, bytes) for every inline data part."""
    for c in resp.get("candidates", []) or []:
        for p in (c.get("content") or {}).get("parts", []) or []:
            d = p.get("inlineData") or p.get("inline_data")
            if d and d.get("data"):
                yield d.get("mimeType") or d.get("mime_type") or "", base64.b64decode(d["data"])


def block_reason(resp: dict) -> str:
    pf = resp.get("promptFeedback") or {}
    if pf.get("blockReason"):
        return pf["blockReason"]
    for c in resp.get("candidates", []) or []:
        fr = c.get("finishReason", "")
        if fr and fr not in ("STOP", "MAX_TOKENS"):
            return fr
    return ""
