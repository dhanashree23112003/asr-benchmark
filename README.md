# ASR Shootout — Bangalore Locality Name Benchmark

Benchmarks three ASR systems on their ability to correctly transcribe Bangalore locality names from natural Hindi/Hinglish conversational speech. Built as an intern assessment for a voice-based blue-collar hiring platform.

**Models evaluated:**
- Deepgram nova-2 (production API baseline)
- faster-whisper large-v3 (open-source, local GPU)
- AI4Bharat IndicWav2Vec (Indian-language specialist, local GPU)

---

## Results (TL;DR)

| Model | Mean WER | Entity Accuracy | Mean Latency |
|---|---|---|---|
| Whisper large-v3 | **0.576** | 60% | 1065ms |
| AI4Bharat IndicWav2Vec | 0.606 | **65%** | **40ms** |
| Deepgram nova-2 | 0.681 | 60% | 524ms |

Full findings in [`report/benchmark_report.md`](report/benchmark_report.md).

---

## Project Structure

```
asr-benchmark/
├── audio/raw/               # 20 .wav recordings (phone mic, 2–5s each)
├── ground_truth/
│   └── ground_truth.json    # Reference transcripts (Devanagari)
├── scripts/
│   ├── common/
│   │   ├── audio_loader.py  # Filename validation and parsing
│   │   ├── metrics.py       # WER, CER, entity hit (fuzzy match)
│   │   └── result_writer.py # CSV writer with duplicate guard
│   ├── models/
│   │   ├── run_deepgram.py       # Deepgram REST API runner
│   │   ├── run_whisper.py        # faster-whisper runner
│   │   └── run_third_model.py    # AI4Bharat HuggingFace runner
│   ├── pipeline.py          # Orchestrator
│   ├── analyze.py           # Aggregation, charts, pivot tables
│   └── generate_gt_template.py  # Scaffolds ground_truth.json from audio filenames
├── results/
│   ├── raw/                 # Per-model CSVs (20 rows each)
│   └── aggregated/          # Summary tables + PNG charts
├── report/
│   └── benchmark_report.md  # 3-page findings report
├── notebooks/
│   └── asr_benchmark.ipynb  # Colab entrypoint
├── requirements.txt
└── .env.example
```

---

## Quickstart (Colab)

**Runtime:** GPU → T4 required for Whisper and AI4Bharat.

**1. Upload this folder to Google Drive** (anywhere inside `MyDrive/`).

**2. Open `notebooks/asr_benchmark.ipynb` in Colab.**  
Set runtime to T4: *Runtime → Change runtime type → T4 GPU*

**3. Add your Deepgram API key as a Colab Secret**  
Left sidebar → Key icon → add `DEEPGRAM_API_KEY`

**4. Run all cells top to bottom.**  
Cell 1 auto-detects the project folder in Drive — no path editing needed.

Expected runtime: ~20 min (Whisper downloads ~3GB on first run, cached after).

---

## Running Locally (API models only, no GPU needed)

```bash
pip install -r requirements.txt
```

```bash
export DEEPGRAM_API_KEY=your_key_here

# Run only API-based models (no GPU required)
python scripts/pipeline.py --models deepgram

# Run analysis
python scripts/analyze.py
```

Whisper and AI4Bharat require a CUDA GPU. Skip them with `--models deepgram` if running on CPU.

---

## Audio Naming Convention

```
{locality}_{condition}_{take}.wav
```

- `locality`: lowercase, hyphens for spaces — e.g. `hsr-layout`, `koramangala`
- `condition`: one of `quiet` / `noisy` / `phone` / `whispered`
- `take`: zero-padded integer — `01`, `02`

Examples: `koramangala_quiet_01.wav`, `hsr-layout_phone_01.wav`

To add new recordings, drop `.wav` files into `audio/raw/`, then:
```bash
python scripts/generate_gt_template.py   # scaffolds ground_truth.json entries
# fill in "transcript" fields, then run pipeline
```

---

## Metrics

| Metric | Definition |
|---|---|
| **WER** | `(S + D + I) / N_ref_words` via jiwer |
| **CER** | Levenshtein distance at character level via jiwer |
| **Entity hit** | `rapidfuzz.fuzz.partial_ratio(locality, transcript) ≥ 75` — checks both Roman and Devanagari forms |
| **Latency** | Wall-clock time of API/inference call only (`time.perf_counter`) |

---

## Environment Variables

| Variable | Required | Used by |
|---|---|---|
| `DEEPGRAM_API_KEY` | Yes | `run_deepgram.py` |
| `WHISPER_MODEL_SIZE` | No (default: `large-v3`) | `run_whisper.py` |
| `USE_BHASHINI_API` | No (default: `0`) | `run_third_model.py` |
| `BHASHINI_API_KEY` | Only if `USE_BHASHINI_API=1` | `run_third_model.py` |

See `.env.example` for a template.

---

## Dependencies

```
faster-whisper>=1.0.0
transformers>=4.35.0
jiwer>=3.0.0
rapidfuzz>=3.0.0
pandas>=2.0.0
matplotlib>=3.7.0
soundfile>=0.12.0
librosa>=0.10.0
requests>=2.28.0
pyctcdecode>=0.5.0
```

PyTorch is pre-installed on Colab — do not add it to requirements.txt.
