"""
ASR Benchmark Pipeline — Orchestrator

Iterates every audio file × every model, computes metrics, writes per-model CSVs.

Usage:
  python scripts/pipeline.py                          # run all 3 models
  python scripts/pipeline.py --models deepgram        # single model
  python scripts/pipeline.py --models deepgram whisper
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Allow running as `python scripts/pipeline.py` from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.common.audio_loader import load_audio_files
from scripts.common.metrics import compute_wer, compute_cer, compute_entity_hit, normalize_transcript
from scripts.common.result_writer import append_row

AUDIO_DIR = ROOT / "audio" / "raw"
GT_FILE = ROOT / "ground_truth" / "ground_truth.json"
RESULTS_DIR = ROOT / "results" / "raw"

ALL_MODELS = ["deepgram", "whisper", "ai4bharat"]


def load_ground_truth() -> dict:
    with GT_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def get_runner(model_name: str):
    if model_name == "deepgram":
        from scripts.models.run_deepgram import transcribe
    elif model_name == "whisper":
        from scripts.models.run_whisper import transcribe
    elif model_name == "ai4bharat":
        from scripts.models.run_third_model import transcribe
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return transcribe


def run_model(model_name: str, audio_files, ground_truth: dict) -> None:
    transcribe = get_runner(model_name)
    csv_path = RESULTS_DIR / f"results_{model_name}.csv"
    samples = ground_truth["samples"]

    print(f"\n=== Running model: {model_name} ===")
    for af in audio_files:
        stem = af.path.stem
        gt_entry = samples[stem]

        metadata = {
            "filename": af.filename,
            "locality": af.locality,
            "condition": af.condition,
            "take": af.take,
            "ground_truth": gt_entry["transcript"],
        }

        print(f"  [{model_name}] {af.filename} ...", end=" ", flush=True)
        result = transcribe(af.path, metadata)

        if result.error:
            print(f"ERROR: {result.error}")
            row = {
                **metadata,
                "model": model_name,
                "transcript": "",
                "transcript_normalized": "",
                "wer": float("nan"),
                "cer": float("nan"),
                "entity_hit": 0,
                "fuzzy_score": 0,
                "latency_ms": 0.0,
                "error": result.error,
            }
        else:
            transcript_norm = normalize_transcript(result.transcript)
            wer = compute_wer(metadata["ground_truth"], result.transcript)
            cer = compute_cer(metadata["ground_truth"], result.transcript)
            entity_hit, fuzzy_score = compute_entity_hit(af.locality, result.transcript)

            print(f"WER={wer:.3f}  entity={'HIT' if entity_hit else 'MISS'}  {result.latency_ms:.0f}ms")
            row = {
                **metadata,
                "model": model_name,
                "transcript": result.transcript,
                "transcript_normalized": transcript_norm,
                "wer": wer,
                "cer": cer,
                "entity_hit": int(entity_hit),
                "fuzzy_score": fuzzy_score,
                "latency_ms": result.latency_ms,
                "error": "",
            }

        append_row(csv_path, row)

    print(f"  Saved → {csv_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="ASR benchmark pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Which models to run (default: all three)",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    audio_files = load_audio_files(AUDIO_DIR, ground_truth)

    if not audio_files:
        print("No valid audio files found. Check audio/raw/ and ground_truth.json.")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s). Running models: {args.models}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        run_model(model_name, audio_files, ground_truth)

    print("\nDone. Run `python scripts/analyze.py` to generate summary tables and charts.")


if __name__ == "__main__":
    main()
