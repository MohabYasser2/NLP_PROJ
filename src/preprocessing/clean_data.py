# -*- coding: utf-8 -*-
"""
Arabic Diacritization - Data Cleaning Module (FIXED, ARABIC-CORRECT)

Keeps ONLY:
- Arabic letters (incl. ٱ Alif Wasla)
- Arabic diacritics (tashkīl)
- Dagger alif
- Arabic digits
- Spaces

Fixes safely (linguistically correct):
- Removes punctuation/symbols by replacing with spaces
- Merges ONLY true clitics when they are standalone tokens:
  (وَ / فَ / بِ / كَ / لِ) + <word>  -> merged
- Fixes split definite-article only when it is truly split:
  (وَ / فَ / بِ / كَ / لِ) + "ال" as a separate token -> merged
- Preserves normal word boundaries (never merges two normal words)
"""

import re
import unicodedata
from typing import List

# ======================================================
# Unicode ranges to KEEP
# ======================================================

# Arabic letters + IMPORTANT extras
ARABIC_LETTERS = r"\u0621-\u064A"   # ء..ي
ALIF_WASLA = r"\u0671"             # ٱ (critical: fixes بِٱلْ... cases)
ARABIC_DIACRITICS = r"\u064B-\u0652"  # ً..ْ
DAGGER_ALIF = r"\u0670"            # ٰ
ARABIC_DIGITS = r"\u0660-\u0669"   # ٠..٩

ALLOWED_RE = re.compile(
    rf"[^{ARABIC_LETTERS}{ALIF_WASLA}{ARABIC_DIACRITICS}{DAGGER_ALIF}{ARABIC_DIGITS}\s]"
)

# ======================================================
# Removal regexes
# ======================================================

TATWEEL_RE = re.compile(r"\u0640")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
QURANIC_MARKS_RE = re.compile(r"[\u06D6-\u06ED]")

MULTI_SPACE_RE = re.compile(r"[ \t]+")  # don't eat newlines inside a line (we clean per-line anyway)

# Arabic presence check (letters only)
ARABIC_LETTER_RE = re.compile(rf"[{ARABIC_LETTERS}{ALIF_WASLA}]")

# ======================================================
# SAFE clitic rules (Arabic-correct)
# ======================================================
# We only merge if the clitic is a standalone token:
#   ^ or whitespace  + (و|ف|ب|ك|ل + optional diacritics) + whitespace + next token
# This merges: "وَ قال" -> "وَقال"
# but will NOT merge: "أبو زيد" or "أهل تهامة" etc.

DIAC = rf"[{ARABIC_DIACRITICS}{DAGGER_ALIF}]*"
CLITIC_TOKEN = rf"(?:[وفبكل]{DIAC})"  # one-letter clitic + its diacritics (if any)

# 1) Merge clitic token with ANY following Arabic word token (safe)
#    مثال: "وَ قَالَ" -> "وَقَالَ"
CLITIC_WORD_MERGE_RE = re.compile(
    rf"(^|\s)({CLITIC_TOKEN})\s+([ {ARABIC_LETTERS}{ALIF_WASLA}{ARABIC_DIACRITICS}{DAGGER_ALIF}]+)",
    re.VERBOSE
)

# 2) Specifically merge clitic token with "ال" when "ال" is split as its own token
#    مثال: "بِ ال" -> "بِال"
CLITIC_AL_MERGE_RE = re.compile(
    rf"(^|\s)({CLITIC_TOKEN})\s+(ال{DIAC})",
    re.VERBOSE
)

# If the definite article lost its alif and became "لْ..." after a clitic:
# "بِ لْحَقِّ" -> "بِٱلْحَقِّ"
MISSING_ALIF_ARTICLE_RE = re.compile(
    rf"(^|\s)({CLITIC_TOKEN})\s+(لْ[{ARABIC_LETTERS}{ALIF_WASLA}{ARABIC_DIACRITICS}{DAGGER_ALIF}]+)"
)

# ======================================================
# Helpers
# ======================================================

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def remove_control_chars(text: str) -> str:
    # Remove control chars (we process per-line so no need to keep \n here)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

# ======================================================
# Line cleaning
# ======================================================

def clean_line(line: str) -> str:
    line = normalize_unicode(line)

    # remove known junk marks
    line = TATWEEL_RE.sub("", line)
    line = ZERO_WIDTH_RE.sub("", line)
    line = QURANIC_MARKS_RE.sub("", line)

    line = remove_control_chars(line)

    # Remove everything except allowed Arabic chars (replace with space)
    line = ALLOWED_RE.sub(" ", line)

    # Normalize spaces
    line = MULTI_SPACE_RE.sub(" ", line).strip()

    if not line:
        return ""

    # ✅ FIRST: fix the exact "clitic + ال" split
    # "وَ ال..." -> "وَال..."
    line = CLITIC_AL_MERGE_RE.sub(r"\1\2\3", line)

    # ✅ THEN: merge clitic token with the following word token
    # "وَ قال" -> "وَقال"
    # Important: this ONLY merges when the clitic is a standalone token.
    line = CLITIC_WORD_MERGE_RE.sub(r"\1\2\3", line)

    # Normalize spaces again (in case substitutions create doubles)
    line = MULTI_SPACE_RE.sub(" ", line).strip()

    # ✅ Repair: clitic + "لْ..." (missing alif-wasla for definite article)
    line = MISSING_ALIF_ARTICLE_RE.sub(r"\1\2ٱ\3", line)

    return line

# ======================================================
# Corpus cleaning
# ======================================================

def clean_corpus(lines: List[str]) -> List[str]:
    cleaned = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        line = clean_line(line)
        if not line:
            continue

        # Keep only lines that contain Arabic letters
        if not ARABIC_LETTER_RE.search(line):
            continue

        cleaned.append(line)

    return cleaned

# ======================================================
# File helpers
# ======================================================

def clean_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = clean_corpus(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")

# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strict Arabic cleaner for diacritization (Arabic-correct clitic handling)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    clean_file(args.input, args.output)
    print(f"✅ Cleaned data saved to: {args.output}")
