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
    """Load submission CSV and return dict of ID -> label"""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_id = int(row['ID'])
            label = int(row['label'])
            labels[char_id] = label
    return labels


def calculate_metrics(gold_labels, pred_labels):
    """Calculate accuracy and DER"""
    # Find common IDs
    common_ids = set(gold_labels.keys()) & set(pred_labels.keys())
    
    if not common_ids:
        print("ERROR: No common character IDs found between files!")
        return 0.0, 1.0
    
    # Count correct predictions and errors
    correct = 0
    errors = 0
    total = len(common_ids)
    
    for char_id in sorted(common_ids):
        gold = gold_labels[char_id]
        pred = pred_labels[char_id]
        
        if gold == pred:
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
    gold_labels = load_submission(args.gold_csv)
    print(f"  ✓ Loaded {len(gold_labels)} labels")
    
    print(f"\nLoading predictions: {args.predicted_csv}")
    pred_labels = load_submission(args.predicted_csv)
    print(f"  ✓ Loaded {len(pred_labels)} labels")
    
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
        print(f"{'ID':<10} {'Gold':<10} {'Predicted':<10}")
        print("-" * 70)
        
        mismatch_count = 0
        for char_id in sorted(gold_labels.keys()):
            if char_id in pred_labels:
                gold = gold_labels[char_id]
                pred = pred_labels[char_id]
                if gold != pred:
                    print(f"{char_id:<10} {gold:<10} {pred:<10}")
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
