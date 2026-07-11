import os
import platform
import subprocess
import tempfile
import time

# Platform-specific fixes (Windows local dev only; Docker/Railway runs Linux).
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline.pipeline import run_pipeline

# Audio must be between 30-45 seconds; uploads capped at ~15MB to prevent abuse.
MIN_DURATION_S = 30
MAX_DURATION_S = 45
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Livo Pronunciation Assessment API")

# Allow requests only from trusted frontend origins; configurable via env var for production.
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")  # CORS whitelist
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Pydantic models define phoneme, word, and response shapes returned to the frontend.
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
    # Write upload to a temp file; delete=False so we control cleanup in finally.
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "File too large")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        del contents  # free memory

        # Convert browser-recorded webm/mp3 to 16kHz mono WAV that the pipeline expects.
        if suffix.lower() not in (".wav",):
            wav_path = _convert_to_wav(tmp_path)
            process_path = wav_path
        else:
            process_path = tmp_path

        duration = _get_duration_seconds(process_path)  # validate length
        if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
            raise HTTPException(
                400,
                f"Audio must be {MIN_DURATION_S}-{MAX_DURATION_S}s long (got {duration:.1f}s).",
            )

        # Run the full ASR → G2P → align → GOP score pipeline and stream back results.
        result = run_pipeline(process_path)  # score audio

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
    finally:  # always cleanup
        # Temp audio files are deleted after every request — no user data is retained.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
