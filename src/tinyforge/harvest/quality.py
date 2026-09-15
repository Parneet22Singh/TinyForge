from __future__ import annotations

import re
from collections import Counter


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def score_document(content: str, html_size: int, duplicate: bool = False) -> dict[str, object]:
    words = re.findall(r"\b[\w'-]+\b", content)
    word_count = len(words)
    lines = [line for line in content.splitlines() if line]
    repeated = sum(count - 1 for count in Counter(lines).values() if count > 1)
    text_density = _clamp(len(content) / max(html_size, 1) * 4)
    length_suitability = _clamp(min(word_count / 80, 1.0))
    duplication = _clamp((repeated / max(len(lines), 1)) + (1.0 if duplicate else 0.0))
    punctuation_anomalies = len(re.findall(r"[!?]{3,}|(.)\1{5,}", content))
    coherence = _clamp(1.0 - punctuation_anomalies / max(word_count / 20, 1))
    score = _clamp(text_density * 0.3 + length_suitability * 0.25 + (1 - duplication) * 0.2 + coherence * 0.25)
    return {
        "score": score,
        "signals": {
            "content_density": text_density,
            "boilerplate_ratio": _clamp(max(0.0, 1 - text_density)),
            "duplication": duplication,
            "text_coherence": coherence,
            "length_suitability": length_suitability,
        },
    }
