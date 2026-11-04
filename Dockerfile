# Demucs vocal removal worker for RunPod serverless (Gilad's karaoke-video skill, 04.11.2026).
# The htdemucs/htdemucs_ft and MMS_FA weights are baked in, so a cold start only has to load them.
# 2.7/cu128 (04.11.2026): 2.4/cu124 failed with "no kernel image is available" on a RunPod GPU; 2.7 covers Ampere..Blackwell
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TORCH_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "demucs==4.0.1" "runpod>=1.7" "numpy<2"
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs'); get_model('htdemucs_ft')"
# forced alignment of the lyrics on the vocals (v2, 04.11.2026)
RUN python -c "import torchaudio; torchaudio.pipelines.MMS_FA.get_model(with_star=True)"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
