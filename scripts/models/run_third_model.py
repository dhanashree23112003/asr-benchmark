"""
AI4Bharat IndicWav2Vec runner (third model).

Uses the HuggingFace transformers checkpoint ai4bharat/indicwav2vec-vakyansh-hi.
This is a CTC model fine-tuned on Indian conversational Hindi speech.

Requirements (added to requirements.txt):
  transformers>=4.35.0
  soundfile>=0.12.0
  librosa>=0.10.0

On Colab T4: model downloads ~400MB on first run, then cached.
No API key needed.

Alternative: If you prefer the hosted Bhashini/Dhruva API instead of local inference,
set env var USE_BHASHINI_API=1 and BHASHINI_API_KEY=<key from bhashini.gov.in>.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "ai4bharat"
# Primary: community Hindi Wav2Vec2 fine-tune (no token needed, ~1.2 GB)
# Fallback: set USE_BHASHINI_API=1 + BHASHINI_API_KEY to use AI4Bharat's
#           hosted Dhruva API — register free at bhashini.gov.in
HF_MODEL = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"

_processor = None
_model = None


def _get_hf_model():
    global _processor, _model
    if _model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch
        # Use basic Wav2Vec2Processor (not WithLM) to avoid kenlm dependency
        _processor = Wav2Vec2Processor.from_pretrained(HF_MODEL)
        _model = Wav2Vec2ForCTC.from_pretrained(HF_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"[ai4bharat] Model loaded on {device}")
    return _processor, _model


def _transcribe_hf(audio_path: Path) -> tuple[str, float]:
    """Local HuggingFace inference. Returns (transcript, latency_ms)."""
    import torch
    import librosa

    processor, model = _get_hf_model()
    device = next(model.parameters()).device

    # Load and resample to 16kHz mono (required by IndicWav2Vec)
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    return transcript.lower().strip(), latency_ms


def _transcribe_bhashini(audio_path: Path) -> tuple[str, float]:
    """
    Bhashini Dhruva API inference. Requires BHASHINI_API_KEY env var.
    Register at: https://bhashini.gov.in/ulca/user/register
    """
    import base64
    import requests

    api_key = os.environ.get("BHASHINI_API_KEY")
    if not api_key:
        raise EnvironmentError("BHASHINI_API_KEY not set")

    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": "hi"},
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_b64}]
        }
    }

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    resp = requests.post(
        "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
        json=payload,
        headers=headers,
        timeout=60,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()

    data = resp.json()
    transcript = (
        data["pipelineResponse"][0]["output"][0]["source"]
    )
    return transcript.strip(), latency_ms


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
    Transcribe via AI4Bharat IndicWav2Vec (HuggingFace local) by default.
    Set USE_BHASHINI_API=1 to use the hosted Dhruva API instead.
    """
    base = dict(
        filename=metadata["filename"],
        locality=metadata["locality"],
        condition=metadata["condition"],
        take=metadata["take"],
        model=MODEL_ID,
        ground_truth=metadata["ground_truth"],
    )

    use_api = os.environ.get("USE_BHASHINI_API", "0") == "1"

    try:
        if use_api:
            transcript, latency_ms = _transcribe_bhashini(audio_path)
        else:
            transcript, latency_ms = _transcribe_hf(audio_path)
        return TranscriptResult(**base, transcript=transcript, latency_ms=latency_ms)

    except Exception as exc:
        return TranscriptResult(**base, transcript="", latency_ms=0.0, error=str(exc))
