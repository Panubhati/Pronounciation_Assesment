"""
Stage 3: Forced alignment.

Given the audio and the expected phoneme sequence (from g2p.py), find:
  (a) which audio frames each expected phoneme occupies, and
  (b) the acoustic model's full posterior distribution at those frames,
      which stage 4 (score.py) needs to compute GOP.

Model: facebook/wav2vec2-lv-60-espeak-cv-ft -- a wav2vec2 checkpoint
fine-tuned with a CTC head over espeak phoneme tokens (not characters,
not words). This is what makes phoneme-level GOP scoring possible at all;
a standard ASR wav2vec2 checkpoint only gives you character posteriors.
"""
from dataclasses import dataclass

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2ForCTC | None = None


def _get_model():
    global _processor, _model
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
        _model.eval()
    return _processor, _model


@dataclass
class PhonemeAlignment:
    phoneme: str
    start_frame: int
    end_frame: int
    # Full log-probability vector over the model's phoneme vocabulary at
    # this phoneme's frames -- needed to see what the model heard instead.
    log_probs: torch.Tensor


@dataclass
class AlignmentResult:
    phonemes: list[PhonemeAlignment]
    emission: torch.Tensor  # [time, vocab] log-probs for the whole clip
    frame_duration_s: float


def align(audio_path: str, expected_phonemes: list[str]) -> AlignmentResult:
    processor, model = _get_model()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # [1, time, vocab]
    emission = torch.log_softmax(logits, dim=-1)[0]  # [time, vocab]

    vocab = processor.tokenizer.get_vocab()
    token_ids = []
    for p in expected_phonemes:
        if p not in vocab:
            # Model didn't see this phoneme in training -- skip rather than
            # crash. Log these in practice; a high skip rate usually means
            # a phonemizer/vocab mismatch worth investigating.
            continue
        token_ids.append(vocab[p])

    if not token_ids:
        return AlignmentResult(phonemes=[], emission=emission, frame_duration_s=waveform.shape[1] / sr / emission.shape[0])

    targets = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    input_lengths = torch.tensor([emission.shape[0]])
    target_lengths = torch.tensor([targets.shape[1]])

    # torchaudio.functional.forced_align (2.1+) runs CTC forced alignment:
    # the best monotonic path through the emission matrix that produces
    # exactly the target token sequence.
    aligned_tokens, scores = torchaudio.functional.forced_align(
        emission.unsqueeze(0), targets, input_lengths, target_lengths, blank=processor.tokenizer.pad_token_id,
    )
    spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    phoneme_alignments = []
    for span, phoneme in zip(spans, expected_phonemes):
        phoneme_alignments.append(
            PhonemeAlignment(
                phoneme=phoneme,
                start_frame=span.start,
                end_frame=span.end,
                log_probs=emission[span.start:span.end].mean(dim=0),
            )
        )

    frame_duration_s = (waveform.shape[1] / sr) / emission.shape[0]
    return AlignmentResult(phonemes=phoneme_alignments, emission=emission, frame_duration_s=frame_duration_s)
