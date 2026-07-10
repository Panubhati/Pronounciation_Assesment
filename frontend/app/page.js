"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const MIN_DURATION = 30;
const MAX_DURATION = 45;
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function scoreColor(score) {
  if (score >= 80) return "var(--good)";
  if (score >= 60) return "var(--caution)";
  return "var(--flagged)";
}

function getOverallFeedback(score) {
  if (score >= 80) {
    return "Strong clarity. Most words should be easy for a listener to understand.";
  }
  if (score >= 60) {
    return "Mostly understandable. Practice the yellow and red words first.";
  }
  return "Needs practice. Slow down, speak a little louder, and repeat the red words.";
}

function getWordFeedback(word) {
  const score = Math.round(word.score);
  const flag = word.flag || "";

  if (flag.includes("low ASR confidence") || flag.includes("unclear segment")) {
    return {
      label: "Unclear audio",
      meaning: "The app could not hear this word confidently.",
      nextStep: "Say it a little louder and reduce background noise.",
    };
  }

  if (score >= 80) {
    return {
      label: "Clear",
      meaning: "This word sounded easy to understand.",
      nextStep: "Keep this pronunciation.",
    };
  }

  if (score >= 60) {
    return {
      label: "Almost clear",
      meaning: "The word was understandable, but one part may be a little soft.",
      nextStep: "Repeat it slowly and keep the ending sound clear.",
    };
  }

  return {
    label: "Try again",
    meaning: "One or more sounds in this word were different from the expected pronunciation.",
    nextStep: "Break the word into smaller parts, then say the full word again.",
  };
}

function getAudioDuration(file) {
  return new Promise((resolve, reject) => {
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(audio.src);
      resolve(audio.duration);
    };
    audio.onerror = () => reject(new Error("Could not read audio metadata"));
    audio.src = URL.createObjectURL(file);
  });
}

export default function Page() {
  const [file, setFile] = useState(null);
  const [duration, setDuration] = useState(null);
  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState("idle");
  const [analysisStep, setAnalysisStep] = useState(0);
  const [analysisTime, setAnalysisTime] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const fileInputRef = useRef(null);
  const timerRef = useRef(null);
  const analysisTimerRef = useRef(null);

  const ANALYSIS_STEPS = [
    "Uploading audio...",
    "Transcribing speech...",
    "Analyzing pronunciation...",
    "Calculating scores...",
  ];

  const handleFilePicked = useCallback(async (picked) => {
    setError(null);
    setResult(null);
    try {
      const dur = await getAudioDuration(picked);
      setFile(picked);
      setDuration(dur);
    } catch (e) {
      setError("That file doesn't look like a readable audio clip. Try a WAV or MP3.");
    }
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    const picked = e.dataTransfer.files?.[0];
    if (picked) handleFilePicked(picked);
  };

  const startRecording = async () => {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const recordedFile = new File([blob], "recording.webm", { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        await handleFilePicked(recordedFile);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      setRecordingTime(0);
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          const next = prev + 1;
          if (next >= MAX_DURATION) {
            mediaRecorderRef.current?.stop();
            setRecording(false);
            clearInterval(timerRef.current);
          }
          return next;
        });
      }, 1000);
    } catch (e) {
      setError("Couldn't access your microphone. You can upload a file instead.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    clearInterval(timerRef.current);
    setRecording(false);
  };

  useEffect(() => () => clearInterval(timerRef.current), []);

  const durationInRange = duration != null && duration >= MIN_DURATION && duration <= MAX_DURATION;
  const canSubmit = file && durationInRange && consent && status !== "uploading";

  const submit = async () => {
    setStatus("uploading");
    setError(null);
    setAnalysisStep(0);
    setAnalysisTime(0);

    analysisTimerRef.current = setInterval(() => setAnalysisTime((t) => t + 1), 1000);

    const stepTimers = [
      setTimeout(() => setAnalysisStep(1), 2000),
      setTimeout(() => setAnalysisStep(2), 8000),
      setTimeout(() => setAnalysisStep(3), 20000),
    ];

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Something went wrong while analyzing your audio.");
      setStatus("idle");
    } finally {
      clearInterval(analysisTimerRef.current);
      stepTimers.forEach(clearTimeout);
    }
  };

  return (
    <main>
      <p className="eyebrow">Pronunciation check</p>
      <h1>How clear was that?</h1>
      <p className="subtitle">
        Upload or record 30-45 seconds of English speech. You'll get a score and see exactly
        which words to work on.
      </p>

      <div className="panel">
        <div
          className="dropzone"
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <div>{file ? file.name : "Drop an audio file here, or click to choose one"}</div>
          <div className="hint">WAV or MP3, 30-45 seconds</div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => e.target.files?.[0] && handleFilePicked(e.target.files[0])}
          />
        </div>

        <div className="record-row">
          {!recording ? (
            <button onClick={startRecording}>Record instead</button>
          ) : (
            <>
              <button className="recording" onClick={stopRecording}>
                Stop recording
              </button>
              <span className="recording-timer">
                {String(Math.floor(recordingTime / 60)).padStart(2, "0")}:{String(recordingTime % 60).padStart(2, "0")}
                <span className="recording-timer-max"> / {MAX_DURATION}s</span>
              </span>
            </>
          )}
          {duration != null && (
            <span className={`duration-readout ${durationInRange ? "" : "out-of-range"}`}>
              {duration.toFixed(1)}s {durationInRange ? "" : `(needs to be ${MIN_DURATION}-${MAX_DURATION}s)`}
            </span>
          )}
        </div>

        <label className="consent-row">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>
            I consent to my audio being processed to generate this score. It's used only for that,
            never stored, and deleted immediately after processing.
          </span>
        </label>

        {error && <div className="error-banner">{error}</div>}

        {status === "uploading" ? (
          <div className="analysis-overlay">
            <div className="analysis-spinner" />
            <div className="analysis-steps">
              {ANALYSIS_STEPS.map((step, i) => (
                <div
                  key={i}
                  className={`analysis-step ${
                    i < analysisStep ? "done" : i === analysisStep ? "active" : ""
                  }`}
                >
                  <span className="step-icon">
                    {i < analysisStep ? "OK" : i === analysisStep ? "..." : ""}
                  </span>
                  {step}
                </div>
              ))}
            </div>
            <div className="analysis-elapsed">
              {Math.floor(analysisTime / 60)}:{String(analysisTime % 60).padStart(2, "0")} elapsed
            </div>
            <div className="analysis-hint">This may take up to 2 minutes on first run</div>
          </div>
        ) : (
          <div style={{ marginTop: 20 }}>
            <button className="primary" disabled={!canSubmit} onClick={submit}>
              Get my score
            </button>
          </div>
        )}
      </div>

      {result && (
        <section className="results">
          <div className="score-meter">
            <span className="value" style={{ color: scoreColor(result.overall_score) }}>
              {result.overall_score}
            </span>
            <span className="label">overall / 100</span>
          </div>
          <p className="result-summary">{getOverallFeedback(result.overall_score)}</p>

          <div className="transcript">
            {result.words.map((w, i) => {
              const feedback = getWordFeedback(w);
              return (
                <span
                  key={i}
                  className={`word ${w.flag || w.score < 60 ? "flagged" : ""}`}
                  tabIndex={0}
                  style={{ borderColor: scoreColor(w.score) }}
                  aria-label={`${w.word}: ${feedback.label}, ${Math.round(w.score)} out of 100`}
                >
                  {w.word}
                  <span className="tooltip">
                    <span className="tooltip-header">
                      <span className="tooltip-word">{w.word}</span>
                      <span className="score-pill" style={{ color: scoreColor(w.score) }}>
                        {Math.round(w.score)}/100
                      </span>
                    </span>
                    <span className="feedback-label">{feedback.label}</span>
                    <span className="feedback-copy">{feedback.meaning}</span>
                    <span className="feedback-next">{feedback.nextStep}</span>
                  </span>
                </span>
              );
            })}
          </div>

          <div className="legend">
            <span><span className="swatch" style={{ background: "var(--good)" }} />Clear (80+)</span>
            <span><span className="swatch" style={{ background: "var(--caution)" }} />Almost clear (60-79)</span>
            <span><span className="swatch" style={{ background: "var(--flagged)" }} />Try again (&lt;60)</span>
          </div>
        </section>
      )}

      <p className="footer-note">
        Your audio is processed in memory to generate this feedback and is deleted immediately
        afterward. Nothing is stored, logged, or shared. See the architecture doc in this repo for
        the full data-handling policy under India's DPDP Act, 2023.
      </p>
    </main>
  );
}
