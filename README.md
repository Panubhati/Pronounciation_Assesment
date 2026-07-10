# Livo Pronunciation Assessment

Upload or record 30-45s of English speech, get an overall pronunciation
score plus word-level (and phoneme-level) feedback on what specifically
went wrong.

See `ARCHITECTURE.md` for the full design writeup (models used, scoring
method, DPDP compliance, trade-offs).

## Project layout

```
backend/    FastAPI service: Whisper -> G2P -> forced alignment -> GOP scoring
frontend/   Next.js app: upload/record UI + results view
```

## Running locally

### Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# phonemizer needs the espeak-ng system binary -- install it separately:
#   macOS:   brew install espeak-ng
#   Ubuntu:  sudo apt-get install espeak-ng
#   Windows: Download & run the .msi from https://github.com/espeak-ng/espeak-ng/releases
#            and configure the environment variable pointing to the DLL:
#            $env:PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"
# (the Docker image below installs this automatically, so this only
# matters if you're running the backend outside Docker)

uvicorn app.main:app --reload --port 8000
```

First request will be slow -- it downloads and loads Whisper (`base.en`)
and the wav2vec2 phoneme model. Subsequent requests reuse the loaded
models (see the singleton pattern in `transcribe.py` / `align.py`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`. It talks to `http://localhost:8000` by
default; override with `NEXT_PUBLIC_API_URL` (see `.env.local.example`)
if your backend runs elsewhere.

## Deploying

**Backend** -- deploy via the included `Dockerfile` (Railway, Fly.io,
Render's Docker service type all work). A plain Python buildpack will
NOT work because `phonemizer` shells out to the `espeak-ng` system
binary, which only the Docker image installs. Set `ALLOWED_ORIGINS` to
your deployed frontend's URL once you have it.

**Frontend** -- deploy to Vercel, pointing `NEXT_PUBLIC_API_URL` at your
deployed backend's URL.

Order matters: deploy the backend first, copy its URL into the
frontend's env var, then deploy the frontend, then go back and set
`ALLOWED_ORIGINS` on the backend to the frontend's real URL (chicken-and
-egg between the two -- expect one redeploy of the backend after the
frontend URL exists).

## Known gaps / what's next

See the "Trade-offs" section of `ARCHITECTURE.md`.
