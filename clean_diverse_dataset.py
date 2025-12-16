"""
Clean train_diverse.txt to only use the 15 valid diacritics.
Normalizes all invalid/conflicting diacritic combinations.
"""
import sys
sys.path.insert(0, '.')
from src.preprocessing.tokenize import tokenize_line
from src.preprocessing.encode_labels import load_diacritic_map
import collections

def normalize_diacritics(diac):
    """Normalize diacritic to one of the 15 valid forms."""
    if not diac:
        return ""
    
    # Remove duplicates
    cleaned = []
    prev = None
    for char in diac:
        if char != prev:
            cleaned.append(char)
        prev = char
    diac = ''.join(cleaned)
    
    # Define categories
    shadda = 'ّ'
    sukun = 'ْ'
    base_vowels = {'َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ'}
    
    # Handle conflicting vowels (keep only first)
    found_vowels = [c for c in diac if c in base_vowels]
    if len(found_vowels) > 1:
        first_vowel = found_vowels[0]
        diac = ''.join([c for c in diac if c not in base_vowels or c == first_vowel])
    
    # Invalid: shadda+sukun or vowel+sukun -> remove sukun
    has_vowel_or_shadda = any(c in base_vowels or c == shadda for c in diac)
    if sukun in diac and has_vowel_or_shadda:
        diac = diac.replace(sukun, '')
    
    # Ensure shadda comes first
    if shadda in diac and len(diac) > 1:
        other_chars = diac.replace(shadda, '')
        if other_chars:
            diac = shadda + other_chars
    
    return diac

# Load valid diacritics
valid_diacs = set(load_diacritic_map('utils/diacritic2id.pickle').keys())
print(f"✓ Valid diacritics ({len(valid_diacs)}): {sorted(valid_diacs)}\n")

# Scan entire dataset
print("Scanning train_diverse.txt for invalid diacritics...")
invalid_diacs = collections.Counter()
total_lines = 0
total_chars = 0

lines = open('data/train_diverse.txt', encoding='utf-8').readlines()

for i, line in enumerate(lines):
    total_lines += 1
    try:
        X, Y = tokenize_line(line)
        total_chars += len(Y)
        for diac in Y:
            if diac not in valid_diacs:
                invalid_diacs[diac] += 1
    except Exception as e:
        print(f"⚠ Error on line {i+1}: {e}")

print(f"\n📊 Dataset Statistics:")
print(f"  Total lines: {total_lines:,}")
print(f"  Total characters: {total_chars:,}")
print(f"  Invalid diacritics found: {len(invalid_diacs)} types, {sum(invalid_diacs.values())} occurrences\n")

if invalid_diacs:
    print(f"❌ Invalid diacritics before normalization:")
    for diac, count in invalid_diacs.most_common(50):
        normalized = normalize_diacritics(diac)
        status = "✅" if normalized in valid_diacs else "❌"
        print(f"  {repr(diac):15} -> {repr(normalized):15} count: {count:5} {status}")
    
    # Clean the file
    print(f"\n🔧 Cleaning dataset...")
    with open('data/train_diverse_clean.txt', 'w', encoding='utf-8') as out:
        for line in lines:
            # Re-diacritize with normalized diacritics
            try:
                X, Y = tokenize_line(line)
                # Normalize diacritics
                Y_normalized = [normalize_diacritics(d) for d in Y]
                
                # Reconstruct line
                result = []
                for char, diac in zip(X, Y_normalized):
                    result.append(char + diac)
                
                out.write(''.join(result) + '\n')
            except Exception as e:
                # If tokenization fails, write original line
                out.write(line)
    
    print(f"✅ Cleaned dataset written to: data/train_diverse_clean.txt")
    
    # Verify
    print(f"\n🔍 Verifying cleaned dataset...")
    still_invalid = collections.Counter()
    for line in open('data/train_diverse_clean.txt', encoding='utf-8'):
        try:
            X, Y = tokenize_line(line)
            for diac in Y:
                if diac not in valid_diacs:
                    still_invalid[diac] += 1
        except:
            pass
    
    if still_invalid:
        print(f"⚠️  Still found {len(still_invalid)} invalid types after cleaning:")
        for diac, count in still_invalid.most_common(20):
            print(f"  {repr(diac):15} count: {count}")
    else:
        print(f"✅ All diacritics now valid!")
        print(f"\n📝 Next step: Replace train_diverse.txt with cleaned version:")
        print(f"   mv data/train_diverse.txt data/train_diverse_backup.txt")
        print(f"   mv data/train_diverse_clean.txt data/train_diverse.txt")
else:
    print("✅ No invalid diacritics found! Dataset is clean.")
