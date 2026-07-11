import gc
from dataclasses import dataclass

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# wav2vec2 model fine-tuned with espeak phoneme CTC head — enables phoneme-level scoring.
_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2ForCTC | None = None


def _load_model():
    """Load and int8-quantize wav2vec2 model (lazy singleton)."""
    global _processor, _model
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
        # Dynamic int8 quantization — compresses Linear layers by ~3-4x.
        _model = torch.quantization.quantize_dynamic(
            _model, {torch.nn.Linear}, dtype=torch.qint8
        )
        _model.eval()
    return _processor, _model


def unload():
    """Free wav2vec2 model from RAM after alignment is done."""
    global _processor, _model
    _processor = None
    _model = None
    gc.collect()


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
    id_to_phoneme: dict[int, str]


def align(audio_path: str, expected_phonemes: list[str]) -> AlignmentResult:
    processor, model = _load_model()

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

    # Build id-to-phoneme vocab map for scoring (returned so pipeline doesn't re-fetch model).
    id_to_phoneme = {v: k for k, v in processor.tokenizer.get_vocab().items()}

    # Only keep phonemes that exist in the model's vocabulary; skip unknown espeak tokens.
    vocab = processor.tokenizer.get_vocab()
    token_ids = []
    for p in expected_phonemes:
        if p not in vocab:
            continue
        token_ids.append(vocab[p])

    if not token_ids:  # no vocab match
        return AlignmentResult(
            phonemes=[], emission=emission,
            frame_duration_s=waveform.shape[1] / sr / emission.shape[0],
            id_to_phoneme=id_to_phoneme,
        )

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
    return AlignmentResult(
        phonemes=phoneme_alignments, emission=emission,
        frame_duration_s=frame_duration_s, id_to_phoneme=id_to_phoneme,
    )
