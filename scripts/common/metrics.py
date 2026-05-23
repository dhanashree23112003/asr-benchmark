"""Pure metric functions: WER, CER, entity hit via fuzzy match. No I/O."""

import re
import string

import jiwer
from rapidfuzz import fuzz

ENTITY_HIT_THRESHOLD = 75

# Devanagari spellings for each locality so fuzzy-match works on Hindi transcripts
_DEVANAGARI_LOCALITIES = {
    "koramangala":   "कोरमंगला",
    "hsr layout":    "एचएसआर लेआउट",
    "banashankari":  "बनशंकरी",
    "bellandur":     "बेलंदूर",
    "bommanahalli":  "बोम्मनहल्ली",
    "btm layout":    "बीटीएम लेआउट",
    "electronic city": "इलेक्ट्रॉनिक सिटी",
    "hebbal":        "हेब्बल",
    "indiranagar":   "इंदिरानगर",
    "jayanagar":     "जयनगर",
    "kr puram":      "केआर पुरम",
    "majestic":      "मैजेस्टिक",
    "marathahalli":  "मराठाहल्ली",
    "peenya":        "पीन्या",
    "rajajinagar":   "राजाजीनगर",
    "sarjapur":      "सरजापुर",
    "silk board":    "सिल्क बोर्ड",
    "whitefield":    "व्हाइटफील्ड",
    "yelahanka":     "येलहंका",
    "yeshwantpur":   "यशवंतपुर",
}

_JIWER_TRANSFORMS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Returns float in [0, inf). 0.0 = perfect match."""
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    try:
        return jiwer.wer(
            reference,
            hypothesis,
            reference_transform=_JIWER_TRANSFORMS,
            hypothesis_transform=_JIWER_TRANSFORMS,
        )
    except Exception:
        return 1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. More robust than WER for Hinglish word-boundary variation."""
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    try:
        return jiwer.cer(_normalize(reference), _normalize(hypothesis))
    except Exception:
        return 1.0


def compute_entity_hit(locality: str, hypothesis: str, threshold: int = ENTITY_HIT_THRESHOLD) -> tuple[bool, int]:
    """
    Returns (entity_hit: bool, fuzzy_score: int).
    Checks both the Roman locality name and its Devanagari equivalent so it
    works whether the transcript is in Latin or Devanagari script.
    """
    norm_locality = _normalize(locality.replace("-", " "))
    norm_hypothesis = _normalize(hypothesis)
    score = fuzz.partial_ratio(norm_locality, norm_hypothesis)

    deva = _DEVANAGARI_LOCALITIES.get(norm_locality)
    if deva:
        deva_score = fuzz.partial_ratio(deva, hypothesis)
        score = max(score, deva_score)

    return score >= threshold, score


def normalize_transcript(text: str) -> str:
    """Normalized form stored in CSV; used for reproducible WER/CER computation."""
    return _normalize(text)
