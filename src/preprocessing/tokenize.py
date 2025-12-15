# -*- coding: utf-8 -*-
"""
Arabic Diacritization - Tokenization Module

This module converts cleaned Arabic text into:
- base character sequence (Arabic letters / digits)
- diacritic labels aligned 1-to-1 with base characters

Each label is a string of diacritics attached to the base letter.
"""

import unicodedata
from typing import List, Tuple


def canonicalize_diacritics(diacs: List[str]) -> str:
    """
    Canonicalize the order of diacritics.
    Shadda (ّ) comes first, then other diacritics.
    """
    if not diacs:
        return ""

    # Separate shadda from others
    shadda = [d for d in diacs if d == '\u0651']  # ّ
    others = [d for d in diacs if d != '\u0651']

    # Shadda first, then others (preserve their order)
    canonical = shadda + others

    return "".join(canonical)

# Arabic diacritics (tashkīl)
DIACRITIC_CHARS = set(chr(c) for c in range(0x064B, 0x0653))  # ً ٌ ٍ َ ُ ِ ْ ّ

# Arabic base letters (explicitly include Alef Wasla ٱ)
def is_arabic_base_letter(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0621 <= cp <= 0x064A  # Arabic letters
        or cp == 0x0671         # ٱ Alef Wasla
    )


def is_arabic_diacritic(ch: str) -> bool:
    # Most Arabic diacritics are combining marks (Mn)
    return ch in DIACRITIC_CHARS or unicodedata.category(ch) == "Mn"


def is_arabic_digit(ch: str) -> bool:
    return 0x0660 <= ord(ch) <= 0x0669  # ٠–٩


# ======================================================
# Tokenization
# ======================================================

def tokenize_line(line: str) -> Tuple[List[str], List[str]]:
    """
    Tokenize one cleaned Arabic line.

    Returns:
        base_chars: list of base characters (letters / digits)
        labels: list of diacritic strings aligned with base_chars
    """
    # Normalize Unicode to canonical form
    line = unicodedata.normalize("NFKC", line)
    line = unicodedata.normalize("NFC", line)

    base_chars: List[str] = []
    labels: List[str] = []

    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        # Skip spaces completely (word boundaries handled implicitly)
        if ch.isspace():
            i += 1
            continue

        # Arabic digit
        if is_arabic_digit(ch):
            base_chars.append(ch)
            labels.append("")  # digits have no diacritics
            i += 1
            continue

        # Arabic base letter
        if is_arabic_base_letter(ch):
            base = ch
            i += 1

            diacs = []
            while i < n and is_arabic_diacritic(line[i]):
                if line[i] in DIACRITIC_CHARS:
                    diacs.append(line[i])
                i += 1

            # Normalize diacritics order
            diac_str = canonicalize_diacritics(diacs)

            base_chars.append(base)
            labels.append(diac_str)
            continue

        # Anything else (should not exist after cleaning)
        i += 1

    return base_chars, labels


# ======================================================
# File-level helpers
# ======================================================

def tokenize_file(input_path: str) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Tokenize a cleaned text file.

    Returns:
        all_base_chars: list of base-char sequences (per line)
        all_labels: list of label sequences (per line)
        all_lines: list of original lines
    """
    all_base_chars = []
    all_labels = []
    all_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            base_chars, labels = tokenize_line(line)

            # Skip lines with no Arabic characters
            if not base_chars:
                continue

            # Safety check: alignment
            assert len(base_chars) == len(labels), "Token-label length mismatch"

            all_lines.append(line)
            all_base_chars.append(base_chars)
            all_labels.append(labels)

    return all_base_chars, all_labels, all_lines


# ======================================================
# Debug / CLI
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize cleaned Arabic text for diacritization"
    )
    parser.add_argument("--input", required=True, help="Cleaned input text file")
    parser.add_argument("--max_lines", type=int, default=5, help="Lines to preview")

    args = parser.parse_args()

    X, Y = tokenize_file(args.input)

    with open("tokenize_output.txt", "w", encoding="utf-8") as f:
        f.write("=== Tokenization Preview ===\n")
        for i in range(min(args.max_lines, len(X))):
            f.write(f"\nLine {i+1}:\n")
            for ch, lab in zip(X[i], Y[i]):
                f.write(f"{ch} -> {repr(lab)}\n")
