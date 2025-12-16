# -*- coding: utf-8 -*-
"""
Arabic Diacritization - Label Encoding Module

Maps diacritic strings to integer IDs using diacritic2id.pickle.
This is the final preprocessing step before feature extraction / modeling.
"""

import pickle
import unicodedata
from typing import List, Dict, Tuple


# ======================================================
# Load diacritic mapping
# ======================================================

def load_diacritic_map(path: str) -> Dict[str, int]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================
# Canonicalization
# ======================================================

def canonicalize_diacritics(diac: str) -> str:
    """
    Normalize diacritic string to canonical Unicode form.
    Handles data errors like duplicate diacritics.
    """
    if not diac:
        return ""
    
    # Remove duplicate consecutive identical diacritics (data errors)
    # E.g., 'ِِ' -> 'ِ', 'ََ' -> 'َ'
    cleaned = []
    prev = None
    for char in diac:
        if char != prev:
            cleaned.append(char)
        prev = char
    diac = ''.join(cleaned)
    
    # Canonicalize shadda combinations (shadda should come first)
    shadda = 'ّ'
    sukun = 'ْ'
    
    if shadda in diac and sukun in diac:
        # Invalid: shadda+sukun combination -> keep only shadda with base vowel if present
        # E.g., 'َّْ' -> 'َّ'
        diac = diac.replace(sukun, '')
    
    # Ensure shadda comes first if present
    if shadda in diac and len(diac) > 1:
        other_chars = diac.replace(shadda, '')
        if other_chars:
            # Put shadda first: 'َّ' not 'َّ'
            diac = shadda + other_chars
    
    return diac


# ======================================================
# Encoding
# ======================================================

def encode_labels(
    labels: List[str],
    diacritic2id: Dict[str, int]
) -> List[int]:
    """
    Encode a list of diacritic strings into integer IDs.

    Args:
        labels: list of diacritic strings (one per base letter)
        diacritic2id: mapping from diacritic string to ID

    Returns:
        List of integer label IDs
    """
    encoded = []

    for d in labels:
        d_norm = canonicalize_diacritics(d)

        if d_norm not in diacritic2id:
            print(f"Unknown diacritic combination: {repr(d_norm)}")
            print(f"Available diacritics: {list(diacritic2id.keys())}")
            raise ValueError(
                f"Unknown diacritic combination: {repr(d_norm)}"
            )

        encoded.append(diacritic2id[d_norm])

    return encoded


# ======================================================
# File-level helper
# ======================================================

def encode_corpus(
    all_labels: List[List[str]],
    diacritic2id: Dict[str, int]
) -> List[List[int]]:
    """
    Encode labels for the entire corpus.

    Args:
        all_labels: list of label sequences (per line)

    Returns:
        List of encoded label sequences
    """
    all_encoded = []

    for labels in all_labels:
        encoded = encode_labels(labels, diacritic2id)
        all_encoded.append(encoded)

    return all_encoded


# ======================================================
# CLI / Debug
# ======================================================

if __name__ == "__main__":
    import argparse
    from tokenize import tokenize_file

    parser = argparse.ArgumentParser(
        description="Encode Arabic diacritic labels into integer IDs"
    )
    parser.add_argument("--input", required=True, help="Cleaned text file")
    parser.add_argument(
        "--diacritic_map",
        default="../../utils/diacritic2id.pickle",
        help="Path to diacritic2id.pickle"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=3,
        help="Lines to preview"
    )

    args = parser.parse_args()

    # Load mapping
    diacritic2id = load_diacritic_map(args.diacritic_map)

    # Tokenize
    X, Y = tokenize_file(args.input)

    # Encode
    Y_enc = encode_corpus(Y, diacritic2id)

    print("=== Label Encoding Preview ===")
    for i in range(min(args.max_lines, len(X))):
        print(f"\nLine {i+1}:")
        for ch, lab, lab_id in zip(X[i], Y[i], Y_enc[i]):
            print(f"{ch} -> {repr(lab)} -> {lab_id}")
