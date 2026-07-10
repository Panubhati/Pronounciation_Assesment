from dataclasses import dataclass

from .align import align
from .g2p import word_to_phonemes
from .score import WordScore, overall_score, score_word
from .transcribe import transcribe


@dataclass
class PronunciationResult:
    overall_score: float
    words: list[WordScore]
    transcript: str


def run_pipeline(audio_path: str) -> PronunciationResult:
    words = transcribe(audio_path)  # STT stage
    if not words:
        return PronunciationResult(overall_score=0.0, words=[], transcript="")

    # Step 2: G2P per word, track phoneme counts to slice alignment later.
    expected_phonemes: list[str] = []
    phoneme_counts: list[int] = []
    for w in words:
        phonemes = word_to_phonemes(w.word)
        expected_phonemes.extend(phonemes)
        phoneme_counts.append(len(phonemes))

    # Step 3: single whole-clip alignment pass.
    alignment = align(audio_path, expected_phonemes)  # CTC alignment
    from .align import _get_model  # local import to reuse the loaded processor
    processor, _ = _get_model()
    id_to_phoneme = {v: k for k, v in processor.tokenizer.get_vocab().items()}  # decode ids

    # Step 4: slice the flat alignment.phonemes list back per word, in
    # order, using phoneme_counts. Words whose G2P produced zero phonemes
    word_scores: list[WordScore] = []
    cursor = 0
    for w, count in zip(words, phoneme_counts):
        word_alignment_slice = alignment.phonemes[cursor: cursor + count]
        cursor += count
        word_scores.append(score_word(w, word_alignment_slice, id_to_phoneme))  # GOP score

    transcript = " ".join(w.word for w in words)  # full text
    return PronunciationResult(
        overall_score=overall_score(word_scores),
        words=word_scores,
        transcript=transcript,
    )
