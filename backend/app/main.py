"""
FastAPI entrypoint.

DPDP-relevant design decisions live here, not in a policy doc:
  - Audio is written to a temp file only for the duration of processing,
    inside a try/finally that deletes it even if the pipeline throws.
  - Nothing is written to a database. No user ID, no history endpoint.
  - We do not log audio bytes or transcripts; only request metadata
    (timing, status code) goes to stdout via uvicorn's default logging.
  - CORS is locked to the deployed frontend origin in production -- see
    ALLOWED_ORIGINS below, set via env var at deploy time.
"""
import os
import subprocess
import tempfile
import time

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, HTTPException, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from app.pipeline.pipeline import run_pipeline

MIN_DURATION_S = 30
MAX_DURATION_S = 45
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB is generous headroom for a 45s clip

app = FastAPI(title="Livo Pronunciation Assessment API")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PhonemeOut(BaseModel):
    phoneme: str
    score: float
    heard_instead: str | None = None


class WordOut(BaseModel):
    word: str
    start: float
    end: float
    score: float
    flag: str | None = None
    phonemes: list[PhonemeOut]


class AnalyzeResponse(BaseModel):
    overall_score: float
    transcript: str
    words: list[WordOut]


def _get_duration_seconds(path: str) -> float:
    """Get audio duration using ffprobe (bundled with ffmpeg)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def _convert_to_wav(src_path: str) -> str:
    """Convert any audio file to 16kHz mono WAV via ffmpeg.

    Returns the path to the new WAV file. Caller is responsible for cleanup.
    """
    wav_path = src_path.rsplit(".", 1)[0] + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm", "audio/ogg"):
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = None
    wav_path = None
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "File too large")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        # Drop our only other reference to the raw bytes as soon as they're
        # on disk for processing -- nothing keeps them in memory longer
        # than necessary.
        del contents

        # Convert non-WAV formats (e.g. browser-recorded .webm) to WAV
        # so torchaudio and the pipeline can handle them.
        if suffix.lower() not in (".wav",):
            wav_path = _convert_to_wav(tmp_path)
            process_path = wav_path
        else:
            process_path = tmp_path

        duration = _get_duration_seconds(process_path)
        if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
            raise HTTPException(
                400,
                f"Audio must be {MIN_DURATION_S}-{MAX_DURATION_S}s long (got {duration:.1f}s).",
            )

        result = run_pipeline(process_path)

        return AnalyzeResponse(
            overall_score=result.overall_score,
            transcript=result.transcript,
            words=[
                WordOut(
                    word=w.word, start=w.start, end=w.end, score=w.score_0_100, flag=w.flag,
                    phonemes=[
                        PhonemeOut(phoneme=p.phoneme, score=p.score_0_100, heard_instead=p.heard_instead)
                        for p in w.phonemes
                    ],
                )
                for w in result.words
            ],
        )
    finally:
        # DPDP: delete the temp audio file regardless of success/failure.
        # This is the entire "retention policy" for raw audio -- it does
        # not outlive the request that processes it.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
