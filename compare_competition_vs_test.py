# -*- coding: utf-8 -*-
"""
Compare competition mode vs test mode accuracy

Takes a sample from golden diacritized file, extracts diacritics,
runs model in competition mode, and compares accuracy.
"""

import sys
import os
import csv
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.tokenize import tokenize_line
import re

# Diacritics pattern
DIACRITICS = re.compile(r'[\u064B-\u065F]')

def remove_diacritics(text):
    """Remove all diacritics from Arabic text"""
    return DIACRITICS.sub('', text)


def extract_golden_diacritics(input_file, num_lines=None):
    """
    Extract diacritics from golden diacritized file.
    
    Returns:
        - undiacritized_lines: list of strings without diacritics
        - golden_csv_data: list of (id, label) tuples
    """
    print(f"Reading golden file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]
        lines = all_lines[:num_lines] if num_lines else all_lines
    
    print(f"Processing {len(lines)} lines...")
    
    # Load diacritic mapping
    diacritic2id_path = "utils/diacritic2id.pickle"
    with open(diacritic2id_path, "rb") as f:
        diacritic2id = pickle.load(f)
    
    undiacritized_lines = []
    golden_csv_data = []
    char_id = 0
    
    for line_idx, line in enumerate(lines):
        # Tokenize to get base chars and diacritics
        base_chars, diacritics = tokenize_line(line)
        
        # Store undiacritized version
        undiacritized = remove_diacritics(line)
        undiacritized_lines.append(undiacritized)
        
        # Convert diacritics to labels
        for char_idx, diac in enumerate(diacritics):
            # Canonicalize diacritic (ensure shadda comes first)
            if diac and 'ّ' in diac and len(diac) > 1:
                # Remove shadda and reinsert at beginning
                diac_without_shadda = diac.replace('ّ', '')
                diac = 'ّ' + diac_without_shadda
            
            # Get label ID
            if diac in diacritic2id:
                label = diacritic2id[diac]
            else:
                # Unknown diacritic - map to empty
                print(f"Warning: Unknown diacritic '{diac}' at line {line_idx}, char {char_idx}")
                label = 0
            
            golden_csv_data.append((char_id, label))
            char_id += 1
    
    return undiacritized_lines, golden_csv_data


def load_prediction_csv(csv_file):
    """Load predictions from competition CSV."""
    predictions = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_id = int(row['ID'])
            label = int(row['label'])
            predictions.append((char_id, label))
    
    return predictions


def calculate_accuracy(golden, predictions):
    """Calculate character-level accuracy."""
    if len(golden) != len(predictions):
        print(f"WARNING: Length mismatch! Golden: {len(golden)}, Predictions: {len(predictions)}")
        min_len = min(len(golden), len(predictions))
    else:
        min_len = len(golden)
    
    correct = 0
    total = 0
    
    # Track errors by diacritic type
    error_breakdown = {}
    
    for i in range(min_len):
        golden_id, golden_label = golden[i]
        pred_id, pred_label = predictions[i]
        
        if golden_id != pred_id:
            print(f"WARNING: ID mismatch at position {i}! Golden: {golden_id}, Pred: {pred_id}")
        
        total += 1
        if golden_label == pred_label:
            correct += 1
        else:
            key = (golden_label, pred_label)
            error_breakdown[key] = error_breakdown.get(key, 0) + 1
    
    accuracy = correct / total if total > 0 else 0.0
    der = 1.0 - accuracy
    
    return accuracy, der, error_breakdown


def main():
    # Configuration - use validation file
    golden_file = r"data\val.txt"
    num_lines = None  # Use all validation lines
    model_name = "arabert_char_bilstm_crf"
    model_path = r"models\best_arabert_char_bilstm_crf (92_fix).pth"
    
    # Output files
    undiacritized_file = "sample/val_undiacritized.txt"
    golden_csv_file = "sample/val_golden.csv"
    prediction_csv_file = "sample/val_predictions.csv"
    
    # Step 1: Extract golden diacritics
    print("\n" + "="*70)
    print("STEP 1: Extract Golden Diacritics")
    print("="*70)
    undiacritized_lines, golden_csv_data = extract_golden_diacritics(golden_file, num_lines)
    
    # Save undiacritized version
    os.makedirs("sample", exist_ok=True)
    with open(undiacritized_file, "w", encoding="utf-8") as f:
        for line in undiacritized_lines:
            f.write(line + "\n")
    
    print(f"✓ Saved undiacritized text to {undiacritized_file}")
    print(f"  Total lines: {len(undiacritized_lines)}")
    print(f"  Total characters: {len(golden_csv_data)}")
    
    # Save golden CSV
    with open(golden_csv_file, "w", encoding="utf-8") as f:
        f.write("ID,label\n")
        for char_id, label in golden_csv_data:
            f.write(f"{char_id},{label}\n")
    
    print(f"✓ Saved golden CSV to {golden_csv_file}")
    
    # Step 2: Run model in competition mode
    print("\n" + "="*70)
    print("STEP 2: Run Model in Competition Mode")
    print("="*70)
    
    cmd = f'python src/test.py --model {model_name} --model_path "{model_path}" --competition --competition_input "{undiacritized_file}" --competition_output "{prediction_csv_file}"'
    print(f"Running: {cmd}")
    
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"ERROR: Model execution failed with exit code {exit_code}")
        return
    
    print(f"✓ Model completed, predictions saved to {prediction_csv_file}")
    
    # Step 3: Compare results
    print("\n" + "="*70)
    print("STEP 3: Compare Golden vs Predictions")
    print("="*70)
    
    predictions = load_prediction_csv(prediction_csv_file)
    
    accuracy, der, error_breakdown = calculate_accuracy(golden_csv_data, predictions)
    
    print(f"\n📊 RESULTS:")
    print(f"  Total characters: {len(golden_csv_data)}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  DER: {der*100:.2f}%")
    print(f"  Correct: {int(accuracy * len(golden_csv_data))}/{len(golden_csv_data)}")
    
    # Show top errors
    if error_breakdown:
        print(f"\n🔍 Top 10 Error Types:")
        sorted_errors = sorted(error_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Load id2diacritic for readable output
        diacritic2id_path = "utils/diacritic2id.pickle"
        with open(diacritic2id_path, "rb") as f:
            diacritic2id = pickle.load(f)
        id2diacritic = {v: k for k, v in diacritic2id.items()}
        
        for (golden_label, pred_label), count in sorted_errors:
            golden_diac = repr(id2diacritic.get(golden_label, '?'))
            pred_diac = repr(id2diacritic.get(pred_label, '?'))
            print(f"  {golden_diac} → {pred_diac}: {count} times")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETED!")
    print("="*70)
    print(f"Golden CSV: {golden_csv_file}")
    print(f"Prediction CSV: {prediction_csv_file}")
    print("="*70)


if __name__ == "__main__":
    main()
