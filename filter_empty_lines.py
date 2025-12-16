"""
Filter out empty lines from training dataset
"""

def filter_empty_lines(input_file, output_file):
    print(f"Filtering empty lines from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"  Original: {len(lines):,} lines")
    
    # Keep only non-empty lines with actual content
    filtered = [
        line for line in lines 
        if line.strip() and len(line.strip()) > 5
    ]
    
    print(f"  Filtered: {len(filtered):,} lines")
    print(f"  Removed: {len(lines) - len(filtered):,} lines")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered)
    
    print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    filter_empty_lines(
        "data/train_modern_mix.txt",
        "data/train_modern_mix.txt"
    )
