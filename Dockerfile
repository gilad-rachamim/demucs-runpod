# Demucs vocal removal worker for RunPod serverless (Gilad's karaoke-video skill, 04.11.2026).
# The htdemucs weights are baked in, so a cold start only has to load them.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TORCH_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "demucs==4.0.1" "runpod>=1.7" "numpy<2"
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
