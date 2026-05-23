"""Loads and validates audio files from audio/raw/. Parses metadata from filenames."""

import re
from dataclasses import dataclass
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<locality>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<condition>quiet|noisy|phone|whispered)_(?P<take>[0-9]{2})\.wav$"
)


@dataclass
class AudioFile:
    path: Path
    filename: str
    locality: str
    condition: str
    take: str


def load_audio_files(audio_dir: Path, ground_truth: dict) -> list[AudioFile]:
    """Return validated AudioFile list for all .wav files in audio_dir."""
    samples = ground_truth.get("samples", {})
    files: list[AudioFile] = []

    for wav in sorted(audio_dir.glob("*.wav")):
        m = FILENAME_RE.match(wav.name)
        if not m:
            print(f"[SKIP] Filename does not match convention: {wav.name}")
            continue

        stem = wav.stem
        if stem not in samples:
            print(f"[SKIP] No ground truth entry for: {stem}")
            continue

        files.append(
            AudioFile(
                path=wav,
                filename=wav.name,
                locality=m.group("locality"),
                condition=m.group("condition"),
                take=m.group("take"),
            )
        )

    return files
