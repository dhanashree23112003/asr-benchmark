"""faster-whisper large-v3 runner. Local inference on GPU (Colab T4)."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "whisper"

# Set to "medium" for faster iteration during development.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")

_model = None  # module-level singleton; loaded once per session


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cuda",
            compute_type="int8_float16",
        )
    return _model


@dataclass
class TranscriptResult:
    filename: str
    locality: str
    condition: str
    take: str
    model: str
    ground_truth: str
    transcript: str
    latency_ms: float
    error: str = ""


def transcribe(audio_path: Path, metadata: dict) -> TranscriptResult:
    """
    Transcribe a single audio file via faster-whisper.

    metadata keys: filename, locality, condition, take, ground_truth
    """
    base = dict(
        filename=metadata["filename"],
        locality=metadata["locality"],
        condition=metadata["condition"],
        take=metadata["take"],
        model=MODEL_ID,
        ground_truth=metadata["ground_truth"],
    )

    try:
        model = _get_model()
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            str(audio_path),
            language="hi",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,   # mark segment as no-speech if prob > 0.6
            compression_ratio_threshold=2.4,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 100,   # keep 100ms padding around detected speech
            },
            word_timestamps=False,
        )
        transcript = " ".join(seg.text.strip() for seg in segments)
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptResult(**base, transcript=transcript, latency_ms=latency_ms)

    except Exception as exc:
        return TranscriptResult(**base, transcript="", latency_ms=0.0, error=str(exc))
