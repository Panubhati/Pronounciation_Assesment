FROM python:3.11-slim

# espeak-ng is a SYSTEM dependency, not a pip package -- phonemizer just
# shells out to it. ffmpeg is needed for audio format conversion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8001
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
