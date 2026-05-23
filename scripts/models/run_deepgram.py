"""Deepgram nova-2 runner. Uses the REST API directly — no SDK version issues."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

MODEL_ID = "deepgram"
API_URL = "https://api.deepgram.com/v1/listen"


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
    base = dict(
        filename=metadata["filename"],
        locality=metadata["locality"],
        condition=metadata["condition"],
        take=metadata["take"],
        model=MODEL_ID,
        ground_truth=metadata["ground_truth"],
    )

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return TranscriptResult(**base, transcript="", latency_ms=0.0,
                                error="DEEPGRAM_API_KEY not set")

    try:
        audio_bytes = audio_path.read_bytes()
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": "nova-2",
            "language": "hi",
            "smart_format": "true",
            "punctuate": "true",
            "filler_words": "false",
        }

        t0 = time.perf_counter()
        resp = requests.post(API_URL, headers=headers, params=params,
                             data=audio_bytes, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        transcript = (resp.json()["results"]["channels"][0]
                      ["alternatives"][0]["transcript"])
        return TranscriptResult(**base, transcript=transcript, latency_ms=latency_ms)

    except Exception as exc:
        return TranscriptResult(**base, transcript="", latency_ms=0.0, error=str(exc))
