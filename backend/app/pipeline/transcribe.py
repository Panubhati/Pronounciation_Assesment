import gc
from dataclasses import dataclass

from faster_whisper import WhisperModel

_MODEL_SIZE = "base.en"
_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    """Load Whisper model into memory (lazy singleton)."""
    global _model
    if _model is None:
        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def unload():
    """Free Whisper model from RAM so the next model can load."""
    global _model
    _model = None
    gc.collect()


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    confidence: float


def transcribe(audio_path: str) -> list[WordResult]:
    model = _load_model()
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        vad_filter=True,
    )

    words: list[WordResult] = []
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            confidence = round(float(w.probability), 3)
            words.append(
                WordResult(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    confidence=round(confidence, 3),
                )
            )
    return words
