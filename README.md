# demucs-runpod

RunPod serverless worker that removes the vocals from a song (Demucs `htdemucs`, GPU).
Built for a private karaoke tool; the image is published to `ghcr.io/gilad-rachamim/demucs-runpod:latest`
by the GitHub Actions workflow on every push to `main`.

Separate: `{"input": {"audio": "<base64 of any audio file>", "model": "htdemucs_ft", "kbps": 128, "shifts": 1, "want_vocals": true}}`
-> adds `"vocals"` (base64 AAC) when `want_vocals`.

Align (forced alignment, torchaudio MMS_FA): `{"input": {"audio": "<base64 vocals>", "pad": 1.0,
"lines": [{"start": 12.3, "end": 15.0, "words": ["...", "..."]}]}}` -> `{"aligned": [[{"start","end","score"}|null, ...]|null, ...]}`
Job output: `{"audio": "<base64 AAC (ADTS) without vocals>", "format": "aac", "sr": 44100, "seconds": 224.4}`
(or `{"error": ...}`).
