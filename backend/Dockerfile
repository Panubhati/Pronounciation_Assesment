FROM python:3.11-slim

# espeak-ng is a SYSTEM dependency, not a pip package -- phonemizer just
# shells out to it. This is the #1 thing that breaks on hosts that only
# give you a Python buildpack (e.g. plain Render "Python" service type).
# That's why this app deploys via Docker rather than a native buildpack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Warm the model cache at build time so the first real request isn't the
# one paying for a multi-hundred-MB download -- trades a slower image
# build for a faster cold start in production.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"
RUN python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; m='facebook/wav2vec2-lv-60-espeak-cv-ft'; Wav2Vec2Processor.from_pretrained(m); Wav2Vec2ForCTC.from_pretrained(m)"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
