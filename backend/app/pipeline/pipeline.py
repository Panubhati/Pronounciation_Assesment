import threading
from dataclasses import dataclass

from .g2p import word_to_phonemes
from .score import WordScore, overall_score, score_word

# Only one request can run the pipeline at a time to prevent OOM from concurrent model loads.
_pipeline_lock = threading.Lock()


@dataclass
class PronunciationResult:
    overall_score: float
    words: list[WordScore]
    transcript: str


def run_pipeline(audio_path: str) -> PronunciationResult:
    with _pipeline_lock:
        return _run_pipeline_locked(audio_path)


def _run_pipeline_locked(audio_path: str) -> PronunciationResult:
    # Step 1: Load Whisper, transcribe, then free it from RAM.
    from .transcribe import transcribe, unload as unload_whisper
    words = transcribe(audio_path)
    unload_whisper()

    if not words:
        return PronunciationResult(overall_score=0.0, words=[], transcript="")

    # Step 2: G2P per word (lightweight, no model needed).
    expected_phonemes: list[str] = []
    phoneme_counts: list[int] = []
    for w in words:
        phonemes = word_to_phonemes(w.word)
        expected_phonemes.extend(phonemes)
        phoneme_counts.append(len(phonemes))

    # Step 3: Load wav2vec2, align, then free it from RAM.
    from .align import align, unload as unload_wav2vec
    alignment = align(audio_path, expected_phonemes)
    id_to_phoneme = alignment.id_to_phoneme
    unload_wav2vec()

    # Step 4: Score each word using the alignment data (no model needed).
    word_scores: list[WordScore] = []
    cursor = 0
    for w, count in zip(words, phoneme_counts):
        word_alignment_slice = alignment.phonemes[cursor: cursor + count]
        cursor += count
        word_scores.append(score_word(w, word_alignment_slice, id_to_phoneme))

    transcript = " ".join(w.word for w in words)
    return PronunciationResult(
        overall_score=overall_score(word_scores),
        words=word_scores,
        transcript=transcript,
    )
