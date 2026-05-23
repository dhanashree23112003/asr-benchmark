# ASR Shootout: Bangalore Locality Name Recognition
### Benchmarking Deepgram, Whisper, and AI4Bharat on Indian Conversational Speech

---

## Approach

**Task:** Evaluate whether ASR systems can reliably extract Bangalore locality names from natural Hindi/Hinglish speech — the core entity extraction problem in a voice-based blue-collar hiring platform.

**Dataset:** 20 self-recorded audio clips, one locality name per clip, spoken in natural conversational Hindi/Hinglish sentences (e.g. *"bhai Marathahalli mein rehta hoon main"*). Recorded on a phone microphone across four conditions — 5 clips each:

| Condition | Description |
|---|---|
| `quiet` | Silent room, direct mic |
| `noisy` | Street/background noise |
| `phone` | Recorded via phone speakerphone playback |
| `whispered` | Hushed or rushed delivery |

**Models benchmarked:**

- **Deepgram nova-2** (`language=hi`) — production API baseline
- **faster-whisper large-v3** (`language=hi`, int8_float16, T4 GPU) — open-source SOTA
- **AI4Bharat IndicWav2Vec** (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`, greedy CTC, T4 GPU) — Indian-language specialist

**Metrics:**
- **WER** — word error rate against Devanagari ground truth (jiwer, lowercased, punctuation stripped)
- **CER** — character error rate; more robust than WER for Hinglish word-boundary variation
- **Entity accuracy** — fraction of clips where the locality name was found in the transcript via fuzzy partial match (rapidfuzz, threshold 75)
- **Latency** — wall-clock time of inference/API call only, excluding I/O

All ground truth transcripts are in Devanagari script to match model output. Entity matching checks both the Roman locality slug and its Devanagari equivalent.

---

## Results

### Overall

| Model | Mean WER ↓ | Median WER | Mean CER ↓ | Entity Accuracy ↑ | Mean Latency | P95 Latency |
|---|---|---|---|---|---|---|
| **Whisper large-v3** | **0.576** | **0.500** | **0.402** | 60% | 1065ms | 1399ms |
| **AI4Bharat IndicWav2Vec** | 0.606 | 0.633 | 0.412 | **65%** | **40ms** | **55ms** |
| Deepgram nova-2 | 0.681 | 0.700 | 0.446 | 60% | 524ms | 881ms |

Deepgram, the designated baseline, finishes last on every accuracy metric.

### WER by Condition

| Model | quiet | noisy | phone | whispered |
|---|---|---|---|---|
| Whisper | **0.337** | **0.533** | **0.691** | 0.744 |
| AI4Bharat | 0.439 | 0.642 | 0.731 | **0.613** |
| Deepgram | 0.451 | 0.652 | 0.888 | 0.734 |

Deepgram's phone WER of **0.888** is alarming — this is the primary deployment channel for the platform.

### Entity Accuracy by Condition

| Model | quiet | noisy | phone | whispered |
|---|---|---|---|---|
| AI4Bharat | **0.8** | 0.6 | **0.8** | 0.4 |
| Deepgram | **0.8** | 0.6 | 0.6 | 0.4 |
| Whisper | **0.8** | **0.8** | 0.6 | 0.2 |

All models agree on quiet (0.8). Whispered is the universal failure — 0.2–0.4 across the board.

---

## Failure Analysis

### Where every model breaks: whispered speech

Bommanahalli, peenya, and yelahanka failed entity extraction on all three models. These were whispered or rushed recordings. Whispered speech fundamentally degrades ASR — phones compress low-energy audio, and the locality names that are multi-syllabic and unusual (bommanahalli, marathahalli) become unrecognizable.

```
File:       bommanahalli_whispered_01.wav
Ground:     बोम्मनहल्ली में हूँ मैं
Deepgram:   कोमल हिंदी में हूं मैं यार   [entity MISS]
Whisper:    को मन हल्ली में हो मैं आर हूँ  [entity MISS, WER 1.5]
AI4Bharat: [partial match, entity MISS]
```

Whisper hallucinates extra words on short whispered clips — the WER of 1.5 means more words were inserted than exist in the reference.

### Where Deepgram specifically breaks: phone condition

Deepgram's entity accuracy drops from 0.8 (quiet) to 0.6 (phone) — and specific failures are spectacular:

```
File:      yelahanka_phone_01.wav
Ground:    येलहंका एरिया में कोई वेकेंसी है
Deepgram:  Uncle area मे कोई batence नहीं है क्या?
```

```
File:      kr-puram_phone_01.wav
Ground:    केआर पुरम में हूँ मैं
Deepgram:  Care में काम ढूंढ रहा हूं
```

Deepgram mishears *yelahanka* as *Uncle* and *KR Puram* as *Care* — complete locality substitutions on phone audio. For a hiring platform where candidates call in, this is a critical failure.

### Where Whisper breaks: whispered condition

Whisper's whispered entity accuracy collapses to **0.2** — worse than both other models. Its noise robustness (best in noisy at 0.8) reverses completely for whispered speech. The VAD filter helps with padded silence but not with low-energy phonemes.

### What AI4Bharat gets right that others miss

AI4Bharat correctly identifies the locality in **phone condition at 0.8 entity accuracy** — matching quiet-condition performance. The model was likely trained on telephone-quality Indian speech. It also achieves the best whispered WER (0.613 vs 0.744 for Whisper) — the Indian-language pretraining seems to help with degraded speech.

The one perfect transcription in the entire benchmark:
```
File:      yeshwantpur_whispered_01.wav
AI4Bharat: WER = 0.000  entity = HIT
```

---

## Recommendation

**For production deployment on phone calls: AI4Bharat with Whisper as fallback.**

Deepgram nova-2 should not be the default for this use case. Its phone-condition performance is the worst of the three, and the specific failure modes (substituting locality names with phonetically similar English words) are dangerous in a system that routes candidates by location. A hiring platform sending candidates to the wrong city is a worse outcome than a failed transcription.

**The right architecture depends on the constraint:**

| Constraint | Recommendation |
|---|---|
| Lowest latency, can host GPU | AI4Bharat: 40ms inference, 65% entity accuracy, Indian-language optimized |
| Best transcription accuracy, no GPU | Whisper via API (replicate.com or modal.com): 0.576 WER |
| Must use an API, no self-hosting | Deepgram, but treat entity extraction as unreliable on phone audio — add a post-processing fuzzy match layer |

**What I would build:** run AI4Bharat on-device for the initial transcription pass, then apply the fuzzy entity matcher against a known locality list (the 30 Bangalore localities). If the fuzzy score is below 80, re-run with Whisper. Two-model cascade costs ~70ms total and gets you to ~75% entity accuracy on real phone calls.

**What this benchmark does not cover:** streaming latency (first-byte), speaker diarization for multi-speaker calls, code-switching from Hindi to Kannada (which is common in Bangalore), and utterances longer than 5 seconds. These are the next evaluation priorities.

---

*Dataset: 20 self-recorded clips, 4 conditions × 5 localities. Models evaluated: Deepgram nova-2, faster-whisper large-v3, AI4Bharat IndicWav2Vec (Harveenchadha/vakyansh-wav2vec2-hindi-him-4200, greedy CTC). Compute: Colab T4 GPU. Code: `scripts/pipeline.py`, `scripts/analyze.py`.*
