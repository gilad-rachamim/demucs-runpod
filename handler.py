"""RunPod serverless handler: song in (base64 mp3/m4a/any ffmpeg input) -> same song without the vocals
(base64 ogg/vorbis). Demucs htdemucs on the GPU. Built 04.11.2026 for Gilad's karaoke-video skill."""
import base64
import os
import subprocess
import tempfile
import traceback

import numpy as np
import runpod
import torch

ERR = None
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    MODEL = get_model("htdemucs")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL.to(DEVICE).eval()
    SR = MODEL.samplerate
except Exception:  # reported through the job answer - worker logs are hard to reach from the PC
    ERR = traceback.format_exc()[-1500:]


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy())


def encode(wav, kbps):
    raw = wav.clamp(-1, 1).T.contiguous().numpy().astype(np.float32).tobytes()
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                          "-c:a", "libvorbis", "-b:a", f"{kbps}k", "-f", "ogg", "-"],
                         input=raw, capture_output=True, check=True).stdout
    return base64.b64encode(out).decode()


def handler(job):
    if ERR:
        return {"error": "worker load failed: " + ERR}
    inp = job["input"]
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(inp["audio"]))
        wav = decode(path)
    except Exception:
        return {"error": "decode failed: " + traceback.format_exc()[-800:]}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    ref = wav.mean(0)
    m, s = ref.mean(), ref.std() + 1e-8
    try:
        with torch.no_grad():
            src = apply_model(MODEL, ((wav - m) / s)[None].to(DEVICE), device=DEVICE, shifts=int(inp.get("shifts", 1)),
                              split=True, overlap=0.25, progress=False)[0].cpu()
    except Exception:
        return {"error": "separation failed: " + traceback.format_exc()[-800:]}
    src = src * s + m
    vocals = src[MODEL.sources.index("vocals")]
    inst = src.sum(0) - vocals
    res = {"sr": SR, "seconds": round(wav.shape[1] / SR, 1), "audio": encode(inst, int(inp.get("kbps", 128)))}
    if inp.get("want_vocals"):
        res["vocals"] = encode(vocals, int(inp.get("kbps", 128)))
    return res


runpod.serverless.start({"handler": handler})
