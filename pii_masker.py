"""
DomesticSafetyNet — Privacy-Preserving NLP / NER PII Masking Module
===================================================================
Provides regex + spaCy NER anonymization for domestic safety texts while:
  1. Preventing double-masking token corruption (e.g., [AADHAAR] -> [[org]]).
  2. Avoiding greedy false-positive address matches (e.g., 'house now', 'C-PTSD',
     or multi-sentence clauses ending in 'have'/'just'/'connect').
  3. Returning structured PII entity detection metadata for privacy metrics.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

try:
    import spacy
except ImportError:
    spacy = None


_NLP_MODEL = None


def get_spacy_nlp():
    """Lazy-load spaCy 'en_core_web_sm' model, downloading if needed in cloud environments."""
    global _NLP_MODEL
    if _NLP_MODEL is not None:
        return _NLP_MODEL
    if spacy is None:
        return None
    try:
        _NLP_MODEL = spacy.load("en_core_web_sm")
    except OSError:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP_MODEL = spacy.load("en_core_web_sm")
        except Exception:
            _NLP_MODEL = None
    return _NLP_MODEL


# Ordered regex patterns for structured PII (Indian + international formats)
REGEX_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # 12-digit Indian Aadhaar number (XXXX XXXX XXXX)
    ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    # Indian PAN card number (5 letters, 4 digits, 1 letter)
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")),
    # Indian vehicle license plate (e.g. MH01AB1234, MP25CE3456)
    ("LICENSE_PLATE", re.compile(r"\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}\b")),
    # Email addresses
    ("EMAIL", re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")),
    # Indian mobile numbers (+91 / 0 prefix optional, starts with 6-9)
    ("PHONE", re.compile(r"(?:\+91[\-\s]?)?[0]?[6789]\d{9}\b")),
    # US / standard phone numbers (e.g. 123-456-7890, (999) 888-7777, 555-0199)
    (
        "PHONE",
        re.compile(
            r"(?:\(\d{3}\)\s*\d{3}[-\.\s]?\d{4}|\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b|\b\d{3}[-\.\s]\d{4}\b)"
        ),
    ),
    # Street addresses (e.g. "123 Elm Street", "42, Marine Drive", "16 Rajpur Road")
    # Uses word boundary before street suffix and restricts to 1-4 tokens to prevent multi-sentence wiping
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}[,\s]+(?:[A-Za-z0-9.]+\s+){1,4}(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Colony|Nagar|Vihar)\b",
            re.IGNORECASE,
        ),
    ),
    # Apartment / Flat / Unit numbers (e.g. "Apartment 4B", "Flat 301", "House No. 10", "C-303", "B-14")
    # Requires a digit in the unit identifier so phrases like "house now" or "C-PTSD" are never falsely masked
    (
        "ADDRESS",
        re.compile(
            r"\b(?:(?:Apartment|Apt\.?|Flat|Flt\.?|House\s+No\.?|H\.?\s*No\.?)\s*#?\s*\d+[A-Za-z0-9\-]*|[CB]-\d+[A-Za-z0-9\-]*)\b",
            re.IGNORECASE,
        ),
    ),
]

NER_TARGET_LABELS = {"PERSON", "GPE", "LOC", "ORG"}


def anonymize_with_details(text: str, use_spacy: bool = True) -> Dict:
    """
    Anonymize text using regex + spaCy NER without token-in-token corruption.

    Returns a dictionary with:
      - 'anonymized_text': Text with [ENTITY] placeholders (original casing)
      - 'cleaned_text': Lowercased, whitespace-normalized text ready for DistilBERT
      - 'entities': List of dicts {'label': str, 'source': 'regex'|'spacy', 'text': str}
      - 'entity_counts': Dict mapping entity label -> count
      - 'latency_ms': Processing time in milliseconds
    """
    start_time = time.perf_counter()
    entities: List[Dict[str, str]] = []
    placeholders: Dict[str, str] = {}
    working_text = text
    counter = 0

    # Step 1: Apply structured regex rules first and replace with inert placeholders
    # (e.g. __PII_0__) that spaCy will not mistake for ORG/GPE entities.
    for label, pattern in REGEX_PII_PATTERNS:
        matches = list(pattern.finditer(working_text))
        for m in reversed(matches):
            matched_str = m.group(0)
            placeholder = f"__PII_{counter}__"
            counter += 1
            placeholders[placeholder] = f"[{label}]"
            entities.append(
                {"label": label, "source": "regex", "text": matched_str}
            )
            working_text = (
                working_text[: m.start()] + placeholder + working_text[m.end() :]
            )

    # Step 2: Run spaCy NER on the remaining text, skipping any existing placeholders
    nlp = get_spacy_nlp() if use_spacy else None
    if nlp is not None:
        doc = nlp(working_text)
        for ent in reversed(doc.ents):
            if ent.label_ in NER_TARGET_LABELS:
                if "__PII_" in ent.text or "[" in ent.text or "]" in ent.text:
                    continue
                entities.append(
                    {"label": ent.label_, "source": "spacy", "text": ent.text}
                )
                working_text = (
                    working_text[: ent.start_char]
                    + f"[{ent.label_}]"
                    + working_text[ent.end_char :]
                )

    # Step 3: Restore regex placeholders to standard [LABEL] tokens
    for placeholder, token in placeholders.items():
        working_text = working_text.replace(placeholder, token)

    # Step 4: Normalize for DistilBERT
    cleaned_text = re.sub(r"\s+", " ", working_text.lower()).strip()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Reverse entities so they appear in reading order
    entities.reverse()
    entity_counts: Dict[str, int] = {}
    for e in entities:
        lbl = e["label"]
        entity_counts[lbl] = entity_counts.get(lbl, 0) + 1

    return {
        "original_text": text,
        "anonymized_text": working_text,
        "cleaned_text": cleaned_text,
        "entities": entities,
        "entity_counts": entity_counts,
        "total_entities": len(entities),
        "latency_ms": round(latency_ms, 3),
    }


def anonymize_text(text: str) -> str:
    """Apply regex + spaCy NER masking to replace PII with bracketed tokens."""
    return anonymize_with_details(text)["anonymized_text"]


def process_with_masking(text: str) -> str:
    """Anonymize + lowercase + whitespace cleanup (WITH NER masking)."""
    return anonymize_with_details(text)["cleaned_text"]


def process_without_masking(text: str) -> str:
    """Lowercase + whitespace cleanup only (NO masking)."""
    return re.sub(r"\s+", " ", text.lower()).strip()

