"""Parse numeric GSM8K answers from model text and compare to gold."""

import re
from typing import Optional, Union


def parse_gsm8k_answer(text: str) -> Optional[Union[int, float]]:
    """Try ``####``, Answer:/answer is, ``\\boxed{}``, then last numeric token."""
    if not text or not text.strip():
        return None

    text = text.strip()

    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return _str_to_number(m.group(1))

    m = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*\$?\s*(-?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return _str_to_number(m.group(1))

    m = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if m:
        return _str_to_number(m.group(1))

    tokens = text.split()
    for token in reversed(tokens):
        if any(c.isdigit() for c in token):
            cleaned = token.replace("$", "").replace(",", "").rstrip(".")
            n = _try_parse_number(cleaned)
            if n is not None:
                return n

    return None


def _str_to_number(s: str) -> Optional[Union[int, float]]:
    s = s.replace(",", "").strip()
    return _try_parse_number(s)


def _try_parse_number(s: str) -> Optional[Union[int, float]]:
    try:
        if "." in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return None


def gsm8k_equal(
    prediction: Optional[Union[int, float]],
    gold: Union[int, float],
    rel_tol: float = 1e-3,
) -> bool:
    """Exact int match or float within ``rel_tol``."""
    if prediction is None:
        return False

    if isinstance(gold, int) and isinstance(prediction, (int, float)):
        return int(round(prediction)) == gold

    if isinstance(gold, float) and isinstance(prediction, (int, float)):
        return abs(float(prediction) - gold) <= rel_tol * max(abs(gold), 1.0)

    return prediction == gold
