import pickle

# Load the arabic_letters.pickle
with open(r'utils\arabic_letters.pickle', 'rb') as f:
    vocab_set = pickle.load(f)

print(f"Type: {type(vocab_set)}")
print(f"Size: {len(vocab_set)}")
print(f"\nLetters ({len(vocab_set)} total):")
for i, letter in enumerate(sorted(list(vocab_set)), 1):
    print(f"{i:2d}. '{letter}' (U+{ord(letter):04X})")

# Now check what CharVocab builds
print("\n" + "="*50)
print("Comparing with CharVocab")
print("="*50)

import sys
sys.path.append('.')
from utils.vocab import CharVocab

# Simulate building vocab from a sample
sample_lines = [
    "اعْلمْ رَحِمَكَ اللهُ",
    "قَالَ الشَّافِعيُّ",
    "123456789"  # Test digits
]

vocab = CharVocab()
for line in sample_lines:
    chars = list(line)
    vocab.build_from_sequences([chars])

print(f"\nCharVocab size: {len(vocab.char2id)}")
print(f"CharVocab contents: {sorted(vocab.char2id.keys())}")

# Find differences
vocab_letters = set(vocab.char2id.keys()) - {'<PAD>', '<UNK>'}
diff_in_charvocab = vocab_letters - vocab_set
diff_in_pickle = vocab_set - vocab_letters

if diff_in_charvocab:
    print(f"\nExtra in CharVocab: {sorted(diff_in_charvocab)}")
if diff_in_pickle:
    print(f"Extra in Pickle: {sorted(diff_in_pickle)}")
