"""
Ground truth template generator.

Scans audio/raw/ for valid .wav files and generates (or updates) ground_truth.json
with skeleton entries. Existing entries are preserved — only missing ones are added.

Usage (run from project root):
  python scripts/generate_gt_template.py

After running, open ground_truth/ground_truth.json and fill in the "transcript"
field for each entry. Leave "notes" blank or add anything useful.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "audio" / "raw"
GT_FILE = ROOT / "ground_truth" / "ground_truth.json"

FILENAME_RE = re.compile(
    r"^(?P<locality>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<condition>quiet|noisy|phone|whispered)_(?P<take>[0-9]{2})\.wav$"
)


def main():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in {AUDIO_DIR}")
        print("Add your recordings there first, then re-run this script.")
        sys.exit(0)

    # Load existing ground truth (preserve already-filled entries)
    if GT_FILE.exists():
        with GT_FILE.open(encoding="utf-8") as f:
            gt = json.load(f)
    else:
        gt = {
            "version": "1.0",
            "normalization_notes": (
                "All transcripts: lowercase, no punctuation, Latin/romanized script, "
                "numbers as words. Be consistent across all samples."
            ),
            "samples": {}
        }

    added = 0
    skipped = 0

    for wav in wavs:
        m = FILENAME_RE.match(wav.name)
        if not m:
            print(f"[SKIP] Does not match naming convention: {wav.name}")
            skipped += 1
            continue

        stem = wav.stem
        if stem in gt["samples"]:
            skipped += 1
            continue

        gt["samples"][stem] = {
            "locality": m.group("locality"),
            "condition": m.group("condition"),
            "take": m.group("take"),
            "transcript": "",   # <-- FILL THIS IN
            "language": "hinglish",  # change to: hindi / kannada / mixed
            "notes": ""
        }
        added += 1

    with GT_FILE.open("w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"ground_truth.json updated.")
    print(f"  Added {added} new skeleton entries.")
    print(f"  Preserved / skipped {skipped} existing or invalid entries.")
    print(f"\nNext: open {GT_FILE} and fill in the 'transcript' field for each entry.")

    # Print a quick checklist of what still needs transcripts
    empty = [k for k, v in gt["samples"].items() if not v.get("transcript", "").strip()]
    if empty:
        print(f"\nEntries still needing transcripts ({len(empty)}):")
        for k in empty:
            print(f"  {k}")
    else:
        print("\nAll entries have transcripts. Ready to run the pipeline.")


if __name__ == "__main__":
    main()
