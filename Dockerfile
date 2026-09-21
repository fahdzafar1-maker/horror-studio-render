FROM python:3.11-slim

# ffmpeg (with libass for captions) + fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-dejavu-core libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Display fonts for thumbnails / Shorts captions (OFL, Google Fonts). Build continues if download fails (DejaVu fallback).
RUN mkdir -p /app/fonts /usr/share/fonts/truetype/hs && \
    (curl -fsSL -o /app/fonts/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf || true) && \
    cp /app/fonts/*.ttf /usr/share/fonts/truetype/hs/ 2>/dev/null || true && fc-cache -f >/dev/null 2>&1 || true

# Pre-download the whisper model at build time so first render does not wait/fail
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')" || true

COPY app ./app

ENV WORK_DIR=/app/storage PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
