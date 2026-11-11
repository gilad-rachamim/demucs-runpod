# Karaoke builder for RunPod serverless (CPU) - Gilad's karaoke bot works when his PC is off (11.11.2026).
# Generic image: no personal code inside. The scripts arrive per job from the Worker (/pc/cloud-code),
# so a script change needs no image rebuild. GPU work (vocal removal, transcription) stays on the GPU endpoints.
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-liberation2 fonts-dejavu-core fontconfig ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Arial (the ASS styles) -> Liberation Sans, which has Hebrew
COPY fonts.conf /etc/fonts/local.conf
RUN pip install --no-cache-dir "runpod>=1.7" "yt-dlp[default]" deno && fc-cache -f
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
