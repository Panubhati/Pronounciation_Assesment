"""
Stage 2: Convert each transcribed word into its expected phoneme sequence.

IMPORTANT: we use `phonemizer` (espeak-ng backend) rather than CMUdict/
g2p_en. The acoustic model used in align.py (wav2vec2-lv-60-espeak-cv-ft)
was trained to output espeak-style IPA phonemes via CTC. If G2P produces
ARPAbet phonemes (CMUdict) while the acoustic model emits espeak phonemes,
every phoneme comparison in GOP scoring silently mismatches. Both stages
must use the same phoneme inventory -- espeak, in this case.

Requires the espeak-ng system binary (see backend Dockerfile).
"""
from phonemizer import phonemize
from phonemizer.separator import Separator

_SEPARATOR = Separator(phone=" ", word="|", syllable="")


def word_to_phonemes(word: str) -> list[str]:
    result = phonemize(
        word,
        language="en-us",
        backend="espeak",
        separator=_SEPARATOR,
        strip=True,
        preserve_punctuation=False,
    )
    return [p for p in result.split(" ") if p]
