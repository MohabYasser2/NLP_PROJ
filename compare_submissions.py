#!/usr/bin/env python3
"""
Compare two competition submission CSV files and calculate metrics.

Usage:
    python compare_submissions.py <gold_csv> <predicted_csv>
    
Example:
    python compare_submissions.py sample/sample_test_set_gold.csv sample/my_submission.csv
"""

import argparse
import csv
import sys


def load_submission(csv_path):
    """Load submission CSV and return dict of ID -> label (and extra info if available)"""
    labels = {}
    has_debug_info = False
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Check if this is a debug format CSV
        has_debug_info = 'letter' in fieldnames and 'predicted_diacritic' in fieldnames
        
        for row in reader:
            # Handle both 'ID' and 'id' column names
            char_id = int(row.get('ID') or row.get('id'))
            label = int(row['label'])
            
            if has_debug_info:
                labels[char_id] = {
                    'label': label,
                    'line_number': int(row.get('line_number', 0)),
                    'letter': row.get('letter', ''),
                    'case_ending': row.get('case_ending', 'False') == 'True',
                    'predicted_diacritic': row.get('predicted_diacritic', '')
                }
            else:
                labels[char_id] = {'label': label}
    
    return labels, has_debug_info


def calculate_metrics(gold_labels, pred_labels):
    """Calculate accuracy and DER"""
    # Find common IDs
    common_ids = set(gold_labels.keys()) & set(pred_labels.keys())
    
    if not common_ids:
        print("ERROR: No common character IDs found between files!")
        return 0.0, 1.0, 0, 0, 0
    
    # Count correct predictions and errors
    correct = 0
    errors = 0
    total = len(common_ids)
    
    for char_id in sorted(common_ids):
        gold_label = gold_labels[char_id]['label'] if isinstance(gold_labels[char_id], dict) else gold_labels[char_id]
        pred_label = pred_labels[char_id]['label'] if isinstance(pred_labels[char_id], dict) else pred_labels[char_id]
        
        if gold_label == pred_label:
            correct += 1
        else:
            errors += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    der = errors / total if total > 0 else 1.0
    
    return accuracy, der, correct, errors, total


def main():
    parser = argparse.ArgumentParser(
        description="Compare two competition CSV submissions and calculate accuracy/DER"
    )
    parser.add_argument(
        "gold_csv",
        help="Path to gold standard CSV file"
    )
    parser.add_argument(
        "predicted_csv",
        help="Path to predicted submission CSV file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed comparison of mismatches"
    )
    
    args = parser.parse_args()
    
    # Load submissions
    print(f"Loading gold standard: {args.gold_csv}")
    gold_labels, gold_has_debug = load_submission(args.gold_csv)
    print(f"  ✓ Loaded {len(gold_labels)} labels")
    
    print(f"\nLoading predictions: {args.predicted_csv}")
    pred_labels, pred_has_debug = load_submission(args.predicted_csv)
    print(f"  ✓ Loaded {len(pred_labels)} labels")
    
    if pred_has_debug:
        print("  ℹ Debug format detected with extra columns")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy, der, correct, errors, total = calculate_metrics(gold_labels, pred_labels)
    
    # Display results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Total characters compared: {total}")
    print(f"Correct predictions:       {correct}")
    print(f"Errors:                    {errors}")
    print(f"\nAccuracy:                  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"DER (Error Rate):          {der:.4f} ({der*100:.2f}%)")
    print("="*70)
    
    # Show detailed mismatches if verbose
    if args.verbose and errors > 0:
        print("\nDetailed mismatches (first 20):")
        print("-" * 70)
        
        if pred_has_debug:
            print(f"{'ID':<6} {'Line':<6} {'Letter':<8} {'Gold':<10} {'Predicted':<10} {'WordEnd':<8}")
            print("-" * 70)
        else:
            print(f"{'ID':<10} {'Gold':<10} {'Predicted':<10}")
            print("-" * 70)
        
        mismatch_count = 0
        for char_id in sorted(gold_labels.keys()):
            if char_id in pred_labels:
                gold_label = gold_labels[char_id]['label'] if isinstance(gold_labels[char_id], dict) else gold_labels[char_id]
                pred_label = pred_labels[char_id]['label'] if isinstance(pred_labels[char_id], dict) else pred_labels[char_id]
                
                if gold_label != pred_label:
                    if pred_has_debug:
                        pred_info = pred_labels[char_id]
                        letter = pred_info.get('letter', '')
                        line_num = pred_info.get('line_number', 0)
                        case_end = pred_info.get('case_ending', False)
                        pred_diac = pred_info.get('predicted_diacritic', '')
                        print(f"{char_id:<6} {line_num:<6} {letter:<8} {gold_label:<10} {pred_label:<10} {str(case_end):<8}")
                        print(f"{char_id:<6} {line_num:<6} {letter:<8} {gold_label:<10} {pred_label:<10} {str(case_end):<8}")
                    else:
                        print(f"{char_id:<10} {gold_label:<10} {pred_label:<10}")
                    
                    mismatch_count += 1
                    if mismatch_count >= 20:
                        if errors > 20:
                            print(f"... and {errors - 20} more mismatches")
                        break
    
    # Check for missing IDs
    gold_only = set(gold_labels.keys()) - set(pred_labels.keys())
    pred_only = set(pred_labels.keys()) - set(gold_labels.keys())
    
    if gold_only:
        print(f"\n⚠ Warning: {len(gold_only)} IDs in gold but not in predictions")
        if len(gold_only) <= 10:
            print(f"  Missing IDs: {sorted(gold_only)}")
    
    if pred_only:
        print(f"\n⚠ Warning: {len(pred_only)} IDs in predictions but not in gold")
        if len(pred_only) <= 10:
            print(f"  Extra IDs: {sorted(pred_only)}")
    
    return 0 if accuracy == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
