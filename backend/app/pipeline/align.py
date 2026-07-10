from dataclasses import dataclass

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# wav2vec2 model fine-tuned with espeak phoneme CTC head — enables phoneme-level scoring.
_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"  # espeak CTC model
_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2ForCTC | None = None


# Lazy singleton: loads the processor and model only once and reuses across all requests.
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
    log_probs: torch.Tensor


@dataclass
class AlignmentResult:
    phonemes: list[PhonemeAlignment]
    emission: torch.Tensor
    frame_duration_s: float


def align(audio_path: str, expected_phonemes: list[str]) -> AlignmentResult:
    processor, model = _get_model()

    waveform, sr = torchaudio.load(audio_path)  # load audio
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)  # resample
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    # Run waveform through wav2vec2 to obtain per-frame phoneme log-probability emission matrix.
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    emission = torch.log_softmax(logits, dim=-1)[0]  # log posteriors

    # Only keep phonemes that exist in the model's vocabulary; skip unknown espeak tokens.
    vocab = processor.tokenizer.get_vocab()
    token_ids = []
    for p in expected_phonemes:
        if p not in vocab:
            continue
        token_ids.append(vocab[p])

    if not token_ids:  # no vocab match
        return AlignmentResult(phonemes=[], emission=emission, frame_duration_s=waveform.shape[1] / sr / emission.shape[0])

    targets = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    input_lengths = torch.tensor([emission.shape[0]])
    target_lengths = torch.tensor([targets.shape[1]])

    # Run CTC forced alignment to find the best monotonic frame-to-phoneme assignment.
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

