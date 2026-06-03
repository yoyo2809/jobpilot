"""Preference parsing helpers shared by the Streamlit UI and tests."""
from __future__ import annotations

import re


def parse_custom_dealbreakers(text: str) -> list[str]:
    """
    Accept either comma-separated keywords or natural-language constraints.
    Examples:
      "No 5+ years ML experience required" -> ["5+ years", "5 years"]
      "No unpaid or contract-only roles" -> ["contract", "unpaid"]
    """
    if not text.strip():
        return []

    lowered = text.lower()
    terms: set[str] = set()

    has_explicit_separators = bool(re.search(r"[,;\n]", text))
    for chunk in re.split(r"[,;\n]+", text):
        cleaned = chunk.strip(" .")
        if cleaned and has_explicit_separators and len(cleaned.split()) <= 5:
            terms.add(cleaned)

    year_matches = re.findall(r"\b([3-9]|1[0-9])\s*\+?\s*(?:years?|yrs?)\b", lowered)
    for year in year_matches:
        terms.add(f"{year}+ years")
        terms.add(f"{year} years")

    phrase_map = {
        "contract": ["contract", "contract-only"],
        "temporary": ["temporary", "temp"],
        "temp": ["temporary", "temp"],
        "1099": ["1099"],
        "unpaid": ["unpaid"],
        "senior": ["senior"],
        "staff": ["staff"],
        "principal": ["principal"],
        "junior": ["junior"],
        "entry": ["entry"],
        "defense": ["defense"],
        "defence": ["defence", "defense"],
        "military": ["military"],
        "production ml": ["production ml", "production machine learning"],
        "production machine learning": ["production ml", "production machine learning"],
        "no sponsorship": ["no sponsorship"],
    }
    for needle, additions in phrase_map.items():
        if needle in lowered:
            terms.update(additions)

    return sorted(terms)
