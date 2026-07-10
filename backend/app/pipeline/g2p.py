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

