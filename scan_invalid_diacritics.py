"""Scan diverse dataset for invalid diacritics."""
import sys
sys.path.insert(0, '.')
from src.preprocessing.tokenize import tokenize_line
from src.preprocessing.encode_labels import load_diacritic_map, canonicalize_diacritics
import collections

# Load valid diacritics
valid_diacs = set(load_diacritic_map('utils/diacritic2id.pickle').keys())
print(f"Valid diacritics ({len(valid_diacs)}): {sorted(valid_diacs)}\n")

# Scan dataset
bad = collections.Counter()
bad_after_canon = collections.Counter()
lines = open('data/train_diverse.txt', encoding='utf-8').readlines()

for i, line in enumerate(lines[:10000]):  # Check first 10K lines
    try:
        X, Y = tokenize_line(line)
        for diac in Y:
            if diac not in valid_diacs:
                bad[diac] += 1
                # Check if canonicalization fixes it
                canon = canonicalize_diacritics(diac)
                if canon not in valid_diacs:
                    bad_after_canon[canon] += 1
    except Exception as e:
        print(f"Error on line {i}: {e}")

if bad:
    print(f"\n❌ Found {len(bad)} invalid diacritic types (BEFORE canonicalization):")
    for diac, count in bad.most_common(50):
        canon = canonicalize_diacritics(diac)
        status = "✅ FIXED" if canon in valid_diacs else "❌ STILL INVALID"
        print(f"  {repr(diac):20} -> {repr(canon):20} count: {count:4} {status}")
else:
    print("✅ No invalid diacritics found!")

if bad_after_canon:
    print(f"\n⚠️  {len(bad_after_canon)} diacritics STILL invalid after canonicalization!")
else:
    print("\n✅ All invalid diacritics fixed by canonicalization!")
