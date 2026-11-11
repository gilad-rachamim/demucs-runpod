"""RunPod serverless handler: build one karaoke song for Gilad's Telegram bot when his PC is off (11.11.2026).

The job input is one bot queue item (the same dict the PC gets from /pc/inbox). The handler:
1. takes the latest scripts from the Worker (GET /pc/cloud-code, a zip of hub_sync.py, karaoke.py and friends),
2. writes the secrets the scripts expect under ~/.claude/secrets from the endpoint's env vars,
3. runs hub_sync.karaoke_turn(item) - exactly what the PC runs - which talks to the Worker over /pc/*.
State (the cache, the .lrc files, karaoke_state.json) lives on the network volume, so a lyrics fix that comes
in a later job finds the song of the earlier one. One worker at a time = one song at a time, like the PC.
Env: HUB_SECRET, RUNPOD_KEY, RUNPOD_TRANSCRIBE (json), RUNPOD_DEMUCS (json), KARAOKE_PROXY, HUB_URL (optional).
"""
import io
import json
import os
import sys
import traceback
import types
import urllib.request
import zipfile

import runpod

VOL = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp/kar"
HOME = os.path.join(VOL, "home")
SKILLS = os.path.join(HOME, ".claude", "skills")
URL = os.environ.get("HUB_URL", "https://gilad-hub.gilad4049.workers.dev")


def setup():
    os.environ["HOME"] = HOME
    sec = os.path.join(HOME, ".claude", "secrets")
    os.makedirs(sec, exist_ok=True)
    files = {"runpod-key.txt": os.environ.get("RUNPOD_KEY", ""),
             "runpod.json": os.environ.get("RUNPOD_TRANSCRIBE", "{}"),
             "runpod-demucs.json": os.environ.get("RUNPOD_DEMUCS", "{}"),
             "telegram-hub.json": json.dumps({"HUB_SECRET": os.environ.get("HUB_SECRET", "")})}
    for name, val in files.items():
        with open(os.path.join(sec, name), "w", encoding="utf-8") as f:
            f.write(val)
    req = urllib.request.Request(URL + "/pc/cloud-code", headers={"User-Agent": "gilad-karaoke-cloud/1.0",
                                                                   "X-Hub-Secret": os.environ.get("HUB_SECRET", "")})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:   # only the scripts are replaced; data/ and the cache stay
        z.extractall(SKILLS)
    for m in ("hub_sync", "karaoke", "runpod_demucs", "runpod_transcribe", "transcribe_calls"):
        sys.modules.pop(m, None)
    sys.path.insert(0, os.path.join(SKILLS, "telegram-hub", "scripts"))
    sys.modules.setdefault("instant_watch", types.ModuleType("instant_watch"))
    import hub_sync
    hub_sync.PY = sys.executable
    hub_sync.shabbat = lambda: False   # the Worker does not start cloud jobs in the Shabbat window
    return hub_sync


def handler(job):
    item = job["input"].get("item") or {}
    try:
        hs = setup()
        if job["input"].get("ping"):
            return {"ok": True, "ping": True, "files": sorted(os.listdir(SKILLS))}
        hs.log(f"cloud job {job.get('id')} item={json.dumps(item, ensure_ascii=False)[:300]}")
        if item.get("bot") == "karaoke" and item.get("act") == "no":
            hs.kf_no(item)
        else:
            hs.karaoke_turn(item)
        return {"ok": True}
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()[-1500:]
        try:
            if item.get("bot") == "karaoke" and not item.get("act"):
                hs.kf_send(item.get("chat"), hs.KF_FAIL)
                hs.kf_done(item.get("chat"), False)
        except Exception:  # noqa: BLE001
            pass
        return {"error": tb}


runpod.serverless.start({"handler": handler})
