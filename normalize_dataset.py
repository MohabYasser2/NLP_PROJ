"""
Normalize train_modern_mix.txt to use the standard 15-diacritic system.
Applies same tokenization/normalization as the original train.txt processing.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.tokenize import tokenize_line
from preprocessing.encode_labels import load_diacritic_map
import unicodedata

def normalize_dataset(input_file, output_file, diacritic_map):
    """
    Read input file, tokenize each line, and save in normalized format.
    Only keeps lines that use the 15 valid diacritics.
    """
    
    print("=" * 80)
    print("Normalizing Dataset to 15-Diacritic System")
    print("=" * 80)
    
    valid_diacritics = set(diacritic_map.keys())
    print(f"\nValid diacritics: {sorted(valid_diacritics)}")
    
    print(f"\n[1] Reading from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"✗ Error reading input: {e}")
        return
    
    print(f"  ✓ Read {len(lines):,} lines")
    
    print(f"\n[2] Tokenizing and normalizing...")
    normalized_lines = []
    skipped_count = 0
    invalid_diacritics = set()
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            # Tokenize to extract base chars and diacritics
            base_chars, diacritic_labels = tokenize_line(line)
            
            # Check if all diacritics are valid
            has_invalid = False
            for diac in diacritic_labels:
                if diac not in valid_diacritics:
                    has_invalid = True
                    invalid_diacritics.add(diac)
            
            if has_invalid:
                skipped_count += 1
                continue
            
            # Reconstruct line with normalized diacritics
            reconstructed = ""
            for base, diac in zip(base_chars, diacritic_labels):
                reconstructed += base + diac
            
            normalized_lines.append(reconstructed + "\n")
            
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1:,}/{len(lines):,} lines...")
        
        except Exception as e:
            skipped_count += 1
            continue
    
    print(f"  ✓ Normalized {len(normalized_lines):,} lines")
    print(f"  ⚠ Skipped {skipped_count:,} lines with invalid diacritics")
    
    if invalid_diacritics:
        print(f"\n  Invalid diacritics found: {sorted(invalid_diacritics)}")
    
    print(f"\n[3] Writing to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(normalized_lines)
        print(f"  ✓ Successfully wrote {len(normalized_lines):,} lines")
    except Exception as e:
        print(f"  ✗ Error writing output: {e}")
        return
    
    print("\n" + "=" * 80)
    print("Normalization Complete")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Total lines: {len(normalized_lines):,}")
    print(f"Skipped: {skipped_count:,}")
    print("=" * 80)

if __name__ == "__main__":
    # Load diacritic mapping
    diacritic_map = load_diacritic_map("utils/diacritic2id.pickle")
    
    # Normalize the dataset
    normalize_dataset(
        input_file="data/train_modern_mix.txt",
        output_file="data/train_modern_mix.txt",  # Overwrite
        diacritic_map=diacritic_map
    )
