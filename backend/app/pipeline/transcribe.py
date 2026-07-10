"""
Stage 1: Speech-to-text with word-level timestamps and confidence.

Uses faster-whisper (CTranslate2 backend) instead of openai-whisper because
it is meaningfully faster on CPU, which matters for a free/cheap deploy
target where you won't have a GPU.
"""
from dataclasses import dataclass
from faster_whisper import WhisperModel

# "base.en" is a good accuracy/speed tradeoff for 30-45s clips on CPU.
# Bump to "small.en" if your host gives you more CPU/RAM headroom and you
# want better accuracy on non-native accents.
_MODEL_SIZE = "base.en"
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Lazy singleton so the model loads once at process start, not per request."""
    global _model
    if _model is None:
        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    confidence: float  # derived from avg_logprob, roughly 0-1


def transcribe(audio_path: str) -> list[WordResult]:
    model = get_model()
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        vad_filter=True,  # trims silence, avoids hallucinated words on empty audio
    )

    words: list[WordResult] = []
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            # faster-whisper already exposes a 0-1 word-level probability.
            # Low values mean the model itself was unsure what it heard --
            # a useful "unclear segment" signal independent of GOP scoring.
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
