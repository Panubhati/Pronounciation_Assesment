"""
Stage 4: Goodness of Pronunciation (GOP) scoring.

For each expected phoneme, the acoustic model gives a full log-probability
distribution over its whole phoneme vocabulary at the aligned frames.
GOP compares "how likely was the expected phoneme" against "how likely
was the single most probable phoneme the model actually thinks it heard."

    GOP(p) = log P(p | audio) - max_j log P(j | audio)

GOP is always <= 0. It equals 0 when the expected phoneme IS the model's
top guess (perfect). It's very negative when the model heard something
else entirely. This is the same core metric used in academic CALL
(computer-assisted language learning) systems and commercial products
like Microsoft's Pronunciation Assessment -- we're not inventing new
math, just implementing a known, defensible one.
"""
from dataclasses import dataclass

import torch

from .align import PhonemeAlignment
from .transcribe import WordResult

# Empirically-tunable: how many log-prob units of "gap" maps to a 0 score.
# Wider = more forgiving. Calibrate this against a small set of native and
# non-native recordings before treating scores as meaningful.
_GOP_SCALE = 5.0


@dataclass
class PhonemeScore:
    phoneme: str
    gop: float
    score_0_100: float
    heard_instead: str | None  # the phoneme the model actually thinks it heard, if different


@dataclass
class WordScore:
    word: str
    start: float
    end: float
    score_0_100: float
    flag: str | None  # None if no issue, else a short human-readable reason
    phonemes: list[PhonemeScore]


def _gop_to_score(gop: float) -> float:
    # gop is <= 0; map to 0-100 with a soft exponential falloff rather than
    # a hard linear clip, so small deviations aren't punished as harshly
    # as completely-wrong phonemes.
    normalized = max(0.0, 1.0 + gop / _GOP_SCALE)
    return round(100 * normalized, 1)


def score_phoneme(pa: PhonemeAlignment, id_to_phoneme: dict[int, str]) -> PhonemeScore:
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
    heard_instead = heard if heard != pa.phoneme else None

    return PhonemeScore(
        phoneme=pa.phoneme,
        gop=round(gop, 3),
        score_0_100=_gop_to_score(gop),
        heard_instead=heard_instead,
    )


def score_word(word: WordResult, word_phoneme_alignments: list[PhonemeAlignment], id_to_phoneme: dict[int, str]) -> WordScore:
    phoneme_scores = [score_phoneme(pa, id_to_phoneme) for pa in word_phoneme_alignments]

    if not phoneme_scores:
        # No phonemes aligned at all -- usually means Whisper transcribed
        # something the acoustic model couldn't find in the audio at all.
        return WordScore(
            word=word.word, start=word.start, end=word.end,
            score_0_100=0.0, flag="unclear segment", phonemes=[],
        )

    # Use the MIN phoneme score, not the average, for the word score.
    # One badly-mispronounced phoneme should flag the whole word even if
    # the other three phonemes in it were fine -- averaging would hide it.
    word_score = min(p.score_0_100 for p in phoneme_scores)

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


def overall_score(word_scores: list[WordScore]) -> float:
    if not word_scores:
        return 0.0
    return round(sum(w.score_0_100 for w in word_scores) / len(word_scores), 1)
