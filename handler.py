"""RunPod serverless handler for Gilad's karaoke-video skill (built 04.11.2026, v2 the same day).

Two jobs:
- separate: song in (base64, any ffmpeg input) -> the song without the vocals (base64 AAC/ADTS), and the vocals
  alone when want_vocals (the PC transcribes the clean voice - far better word times than the full mix).
  Demucs htdemucs_ft by default (better separation, ~4x the GPU seconds of htdemucs - still ~1 cent a song).
- align: the vocals (base64) + lyric lines with rough times -> each word's start/end, by forced alignment
  (torchaudio MMS_FA, wav2vec2 on romanized text - works for Hebrew and Latin-script songs alike).
"""
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
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # reported through the job answer - worker logs are hard to reach from the PC
    ERR = traceback.format_exc()[-1500:]

MODELS = {}
FA = {}


def demucs_model(name):
    if name not in MODELS:
        MODELS[name] = get_model(name).to(DEVICE).eval()
    return MODELS[name]


def fa():
    if not FA:
        b = torchaudio.pipelines.MMS_FA
        FA["model"] = b.get_model(with_star=True).to(DEVICE).eval()
        FA["tok"] = b.get_tokenizer()
        FA["aln"] = b.get_aligner()
        FA["sr"] = b.sample_rate
    return FA


def ffmpeg(args, data=None):
    p = subprocess.run(["ffmpeg", "-v", "error"] + args, input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr.decode("utf-8", "replace")[-600:])
    return p.stdout


def decode(b64, sr, ch):
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(b64))
        raw = ffmpeg(["-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"])
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(-1, ch).T.copy())


def encode(wav, sr, kbps):
    """Native AAC in an ADTS stream: present in every ffmpeg build (the image's conda ffmpeg has no libvorbis)."""
    raw = wav.clamp(-1, 1).T.contiguous().numpy().astype(np.float32).tobytes()
    out = ffmpeg(["-f", "f32le", "-ac", str(wav.shape[0]), "-ar", str(sr), "-i", "-", "-c:a", "aac",
                  "-b:a", f"{kbps}k", "-f", "adts", "-"], raw)
    return base64.b64encode(out).decode()


def separate(inp):
    model = demucs_model(inp.get("model") or "htdemucs_ft")
    sr = model.samplerate
    wav = decode(inp["audio"], sr, 2)
    ref = wav.mean(0)
    m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, ((wav - m) / s)[None].to(DEVICE), device=DEVICE,
                          shifts=int(inp.get("shifts", 1)), split=True, overlap=0.25, progress=False)[0].cpu()
    src = src * s + m
    vocals = src[model.sources.index("vocals")]
    inst = src.sum(0) - vocals
    kbps = int(inp.get("kbps", 128))
    res = {"sr": sr, "seconds": round(wav.shape[1] / sr, 1), "format": "aac", "audio": encode(inst, sr, kbps)}
    if inp.get("want_vocals"):
        res["vocals"] = encode(vocals, sr, kbps)
    return res


# Hebrew -> the Latin letters MMS_FA knows (uroman-like consonants; niqqud and punctuation are dropped)
HEB = dict(zip("אבגדהוזחטיכךלמםנןסעפףצץקרשת",
               ["", "b", "g", "d", "h", "v", "z", "kh", "t", "y", "k", "kh", "l", "m", "m", "n", "n", "s", "",
                "p", "f", "ts", "ts", "k", "r", "sh", "t"]))


def roman(word):
    out = "".join(HEB.get(c, c) for c in word.lower())
    return "".join(c for c in out if "a" <= c <= "z" or c == "'")


def align(inp):
    f = fa()
    sr = f["sr"]
    wav = decode(inp["audio"], sr, 1)
    pad = float(inp.get("pad", 1.0))
    out = []
    for line in inp["lines"]:
        a = max(0, int((float(line["start"]) - pad) * sr))
        b = min(wav.shape[1], int((float(line["end"]) + pad) * sr))
        words = [roman(w) for w in line["words"]]
        idx = [i for i, w in enumerate(words) if w]
        if not idx or b - a < sr // 2:
            out.append(None)
            continue
        try:
            with torch.inference_mode():
                em, _ = f["model"](wav[:, a:b].to(DEVICE))
            spans = f["aln"](em[0], f["tok"](["*"] + [words[i] for i in idx] + ["*"]))[1:-1]
        except Exception:  # too little audio for the letters, etc. - the PC keeps this line's old times
            out.append(None)
            continue
        ratio = (b - a) / em.size(1) / sr
        res = [None] * len(words)
        for i, sp in zip(idx, spans):
            res[i] = {"start": round(a / sr + sp[0].start * ratio, 2), "end": round(a / sr + sp[-1].end * ratio, 2),
                      "score": round(float(sum(t.score for t in sp) / len(sp)), 3)}
        out.append(res)
    return {"aligned": out}


def handler(job):
    if ERR:
        return {"error": "worker load failed: " + ERR}
    inp = job["input"]
    try:
        return align(inp) if inp.get("lines") is not None else separate(inp)
    except Exception:
        return {"error": "job failed: " + traceback.format_exc()[-800:]}


runpod.serverless.start({"handler": handler})
