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

EXPOSE 8001
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
