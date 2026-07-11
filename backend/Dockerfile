FROM python:3.11-slim

# espeak-ng is a SYSTEM dependency for phonemizer. ffmpeg handles audio format conversion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (x86_64 builds default to CUDA otherwise).
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchaudio==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements (torch/torchaudio already installed above).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8001
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
