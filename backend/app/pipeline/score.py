from dataclasses import dataclass

import torch

from .align import PhonemeAlignment
from .transcribe import WordResult

# Controls score sensitivity: larger value = more forgiving for slight deviations.
_GOP_SCALE = 5.0  # tunable sensitivity


@dataclass
class PhonemeScore:
    phoneme: str
    gop: float
    score_0_100: float
    heard_instead: str | None


@dataclass
class WordScore:
    word: str
    start: float
    end: float
    score_0_100: float
    flag: str | None
    phonemes: list[PhonemeScore]


def _gop_to_score(gop: float) -> float:
    # GOP is <= 0; map to 0-100 using soft falloff so minor errors aren't over-penalized.
    normalized = max(0.0, 1.0 + gop / _GOP_SCALE)
    return round(100 * normalized, 1)


def score_phoneme(pa: PhonemeAlignment, id_to_phoneme: dict[int, str]) -> PhonemeScore:
    # Compare log-probability of expected phoneme vs the model's top prediction to compute GOP.
    log_probs = pa.log_probs
    best_id = int(torch.argmax(log_probs).item())
    best_logprob = float(log_probs[best_id].item())

    expected_id = None
    for idx, ph in id_to_phoneme.items():
        if ph == pa.phoneme:
            expected_id = idx
            break

    expected_logprob = float(log_probs[expected_id].item()) if expected_id is not None else best_logprob
    gop = expected_logprob - best_logprob

    heard = id_to_phoneme.get(best_id)
    heard_instead = heard if heard != pa.phoneme else None  # mismatch only

    return PhonemeScore(
        phoneme=pa.phoneme,
        gop=round(gop, 3),
        score_0_100=_gop_to_score(gop),
        heard_instead=heard_instead,
    )


def score_word(word: WordResult, word_phoneme_alignments: list[PhonemeAlignment], id_to_phoneme: dict[int, str]) -> WordScore:
    # Score each phoneme in the word, then take the minimum as the word-level score.
    phoneme_scores = [score_phoneme(pa, id_to_phoneme) for pa in word_phoneme_alignments]

    if not phoneme_scores:
        return WordScore(
            word=word.word, start=word.start, end=word.end,
            score_0_100=0.0, flag="unclear segment", phonemes=[],
        )

    word_score = min(p.score_0_100 for p in phoneme_scores)  # worst phoneme

    # Flag the word if ASR was uncertain or if the worst phoneme score is below 60.
    flag = None
    if word.confidence < 0.5:
        flag = "unclear segment (low ASR confidence)"
    else:
        worst = min(phoneme_scores, key=lambda p: p.score_0_100)
        if worst.score_0_100 < 60:
            if worst.heard_instead:
                flag = f"mispronounced /{worst.phoneme}/ (sounded like /{worst.heard_instead}/)"
            else:
                flag = f"mispronounced /{worst.phoneme}/"

    return WordScore(
        word=word.word, start=word.start, end=word.end,
        score_0_100=word_score, flag=flag, phonemes=phoneme_scores,
    )


# Average all word scores to produce the final session-level pronunciation score.
def overall_score(word_scores: list[WordScore]) -> float:
    if not word_scores:
        return 0.0
    return round(sum(w.score_0_100 for w in word_scores) / len(word_scores), 1)

