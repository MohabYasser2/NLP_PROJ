"""Filter out empty/problematic lines from train_diverse.txt"""
import sys
sys.path.insert(0, '.')
from src.preprocessing.tokenize import tokenize_line

lines = open('data/train_diverse.txt', encoding='utf-8').readlines()
print(f"Original lines: {len(lines)}")

valid_lines = []
empty_count = 0
short_count = 0
error_count = 0

for i, line in enumerate(lines):
    # Check raw length
    if len(line.strip()) < 5:
        empty_count += 1
        continue
    
    # Try tokenization
    try:
        X, Y = tokenize_line(line)
        if len(X) == 0:
            empty_count += 1
            continue
        if len(X) < 3:
            short_count += 1
            continue
        valid_lines.append(line)
    except Exception as e:
        error_count += 1
        print(f"Line {i+1}: {e}")

print(f"Valid lines: {len(valid_lines)}")
print(f"Empty: {empty_count}")
print(f"Too short (<3 chars): {short_count}")
print(f"Errors: {error_count}")

# Write filtered dataset
with open('data/train_diverse.txt', 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

print(f"\n✅ Filtered dataset written: {len(valid_lines)} lines")
