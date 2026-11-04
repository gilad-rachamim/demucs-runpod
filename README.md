# demucs-runpod

RunPod serverless worker that removes the vocals from a song (Demucs `htdemucs`, GPU).
Built for a private karaoke tool; the image is published to `ghcr.io/gilad-rachamim/demucs-runpod:latest`
by the GitHub Actions workflow on every push to `main`.

Job input: `{"input": {"audio": "<base64 of any audio file>", "kbps": 128, "shifts": 1, "want_vocals": false}}`
Job output: `{"audio": "<base64 AAC (ADTS) without vocals>", "format": "aac", "sr": 44100, "seconds": 224.4}`
(or `{"error": ...}`).
