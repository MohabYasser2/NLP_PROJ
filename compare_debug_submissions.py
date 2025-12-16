#!/usr/bin/env python3
"""
Compare debug submission CSV with TA debug CSV to analyze predictions.

Usage:
    python compare_debug_submissions.py <ta_debug_csv> <my_debug_csv> [--gold_csv]
    
Example:
    python compare_debug_submissions.py sample/TA_Debug.csv sample/my_submission_debug.csv
    python compare_debug_submissions.py sample/TA_Debug.csv sample/my_submission_debug.csv --gold_csv sample/gold_labels.csv
"""

import argparse
import csv
import sys
from collections import defaultdict


def load_ta_debug(csv_path):
    """Load TA debug CSV (id, line_number, letter, case_ending)"""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_id = int(row['id'])
            data[char_id] = {
                'line_number': int(row['line_number']),
                'letter': row['letter'],
                'case_ending': row['case_ending'] == 'True'
            }
    return data


def load_my_debug(csv_path):
    """Load my debug CSV (id, line_number, letter, case_ending, label, predicted_diacritic)"""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_id = int(row['id'])
            data[char_id] = {
                'line_number': int(row['line_number']),
                'letter': row['letter'],
                'case_ending': row['case_ending'] == 'True',
                'label': int(row['label']),
                'predicted_diacritic': row['predicted_diacritic']
            }
    return data


def load_gold_labels(csv_path):
    """Load gold labels CSV (ID, label)"""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_id = int(row.get('ID') or row.get('id'))
            labels[char_id] = int(row['label'])
    return labels


def compare_submissions(ta_data, my_data, gold_labels=None):
    """Compare submissions and return analysis"""
    common_ids = set(ta_data.keys()) & set(my_data.keys())
    
    if not common_ids:
        print("ERROR: No common character IDs found!")
        return None
    
    # Check alignment
    misaligned = []
    for char_id in sorted(common_ids)[:100]:  # Check first 100
        ta = ta_data[char_id]
        my = my_data[char_id]
        
        if ta['letter'] != my['letter']:
            misaligned.append({
                'id': char_id,
                'ta_letter': ta['letter'],
                'my_letter': my['letter'],
                'ta_line': ta['line_number'],
                'my_line': my['line_number']
            })
    
    # Calculate statistics
    stats = {
        'total_chars': len(common_ids),
        'case_ending_count': sum(1 for cid in common_ids if my_data[cid]['case_ending']),
        'misaligned': misaligned
    }
    
    # If gold labels provided, calculate accuracy
    if gold_labels:
        correct = 0
        errors = 0
        errors_by_type = defaultdict(int)
        errors_at_word_end = 0
        errors_not_word_end = 0
        
        error_details = []
        
        for char_id in common_ids:
            if char_id in gold_labels:
                gold = gold_labels[char_id]
                pred = my_data[char_id]['label']
                
                if gold == pred:
                    correct += 1
                else:
                    errors += 1
                    is_word_end = my_data[char_id]['case_ending']
                    
                    if is_word_end:
                        errors_at_word_end += 1
                    else:
                        errors_not_word_end += 1
                    
                    # Track error types
                    error_type = f"Gold:{gold} → Pred:{pred}"
                    errors_by_type[error_type] += 1
                    
                    error_details.append({
                        'id': char_id,
                        'line': my_data[char_id]['line_number'],
                        'letter': my_data[char_id]['letter'],
                        'case_ending': is_word_end,
                        'gold_label': gold,
                        'pred_label': pred,
                        'pred_diacritic': my_data[char_id]['predicted_diacritic']
                    })
        
        total = correct + errors
        accuracy = correct / total if total > 0 else 0
        der = errors / total if total > 0 else 0
        
        stats['accuracy'] = accuracy
        stats['der'] = der
        stats['correct'] = correct
        stats['errors'] = errors
        stats['errors_at_word_end'] = errors_at_word_end
        stats['errors_not_word_end'] = errors_not_word_end
        stats['errors_by_type'] = dict(sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True))
        stats['error_details'] = error_details
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare debug submission CSV with TA debug CSV"
    )
    parser.add_argument(
        "ta_debug_csv",
        help="Path to TA debug CSV file"
    )
    parser.add_argument(
        "my_debug_csv",
        help="Path to my debug submission CSV file"
    )
    parser.add_argument(
        "--gold_csv",
        default=None,
        help="Path to gold labels CSV (optional, for accuracy calculation)"
    )
    parser.add_argument(
        "--show_errors",
        type=int,
        default=20,
        help="Number of error examples to show (default: 20)"
    )
    parser.add_argument(
        "--export_errors",
        default=None,
        help="Export all errors to CSV file"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading TA debug: {args.ta_debug_csv}")
    ta_data = load_ta_debug(args.ta_debug_csv)
    print(f"  ✓ Loaded {len(ta_data)} characters")
    
    print(f"\nLoading my debug: {args.my_debug_csv}")
    my_data = load_my_debug(args.my_debug_csv)
    print(f"  ✓ Loaded {len(my_data)} characters")
    
    gold_labels = None
    if args.gold_csv:
        print(f"\nLoading gold labels: {args.gold_csv}")
        gold_labels = load_gold_labels(args.gold_csv)
        print(f"  ✓ Loaded {len(gold_labels)} labels")
    
    # Compare
    print("\nAnalyzing submissions...")
    stats = compare_submissions(ta_data, my_data, gold_labels)
    
    if not stats:
        return 1
    
    # Display results
    print("\n" + "="*70)
    print("SUBMISSION ANALYSIS")
    print("="*70)
    print(f"Total characters:          {stats['total_chars']}")
    print(f"Case ending positions:     {stats['case_ending_count']} ({stats['case_ending_count']/stats['total_chars']*100:.2f}%)")
    
    # Check alignment
    if stats['misaligned']:
        print(f"\n⚠ WARNING: {len(stats['misaligned'])} character misalignments detected!")
        print("First 5 misalignments:")
        for mis in stats['misaligned'][:5]:
            print(f"  ID {mis['id']}: TA='{mis['ta_letter']}' (line {mis['ta_line']}) vs My='{mis['my_letter']}' (line {mis['my_line']})")
    else:
        print("\n✓ Character alignment verified (first 100 samples)")
    
    # Accuracy stats
    if gold_labels:
        print("\n" + "="*70)
        print("ACCURACY METRICS")
        print("="*70)
        print(f"Correct predictions:       {stats['correct']}")
        print(f"Errors:                    {stats['errors']}")
        print(f"Accuracy:                  {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
        print(f"DER (Error Rate):          {stats['der']:.4f} ({stats['der']*100:.2f}%)")
        
        print("\n" + "-"*70)
        print("ERROR LOCATION ANALYSIS")
        print("-"*70)
        print(f"Errors at word endings:    {stats['errors_at_word_end']} ({stats['errors_at_word_end']/stats['errors']*100:.2f}% of errors)")
        print(f"Errors mid-word:           {stats['errors_not_word_end']} ({stats['errors_not_word_end']/stats['errors']*100:.2f}% of errors)")
        
        print("\n" + "-"*70)
        print(f"TOP ERROR TYPES (showing top 10)")
        print("-"*70)
        for i, (error_type, count) in enumerate(list(stats['errors_by_type'].items())[:10], 1):
            print(f"{i:2}. {error_type:<30} {count:>5} times ({count/stats['errors']*100:.2f}%)")
        
        # Show error examples
        if stats['error_details']:
            print("\n" + "-"*70)
            print(f"ERROR EXAMPLES (showing first {min(args.show_errors, len(stats['error_details']))})")
            print("-"*70)
            print(f"{'ID':<8} {'Line':<6} {'Letter':<8} {'WordEnd':<8} {'Gold':<6} {'Pred':<6} {'PredDiac':<10}")
            print("-"*70)
            
            for error in stats['error_details'][:args.show_errors]:
                print(f"{error['id']:<8} {error['line']:<6} {error['letter']:<8} {str(error['case_ending']):<8} "
                      f"{error['gold_label']:<6} {error['pred_label']:<6} {error['pred_diacritic']:<10}")
        
        # Export errors if requested
        if args.export_errors:
            print(f"\nExporting all errors to {args.export_errors}...")
            with open(args.export_errors, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'line', 'letter', 'case_ending', 
                                                       'gold_label', 'pred_label', 'pred_diacritic'])
                writer.writeheader()
                writer.writerows(stats['error_details'])
            print(f"  ✓ Exported {len(stats['error_details'])} errors")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
