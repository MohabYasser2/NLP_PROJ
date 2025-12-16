"""
Concatenate modern books from كتب حديثة folder with subset of train.txt
"""

import os
from pathlib import Path

def concatenate_datasets():
    """
    Concatenate all lines from مكتب حديثة folder with 25k lines from train.txt
    """
    
    print("=" * 80)
    print("Concatenating Modern Books + Train Subset")
    print("=" * 80)
    
    all_lines = []
    
    # 1. Read all files from كتب حديثة folder
    modern_books_dir = Path("texts.txt/msa/كتب حديثة")
    print(f"\n[1] Reading modern books from {modern_books_dir}...")
    
    txt_files = sorted(modern_books_dir.glob("*.txt"))
    
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter empty lines and very short lines
            filtered = [
                line.strip() + '\n' 
                for line in lines 
                if line.strip() and len(line.strip()) > 15
            ]
            
            all_lines.extend(filtered)
            print(f"  ✓ {filepath.name}: {len(filtered)} lines")
        
        except Exception as e:
            print(f"  ✗ Error reading {filepath.name}: {e}")
    
    modern_count = len(all_lines)
    print(f"\n  Total modern books lines: {modern_count}")
    
    # 2. Add مشكل الآثار.txt
    print(f"\n[2] Reading مشكل الآثار.txt...")
    mushkil_path = Path("texts.txt/مشكل الآثار.txt")
    
    if mushkil_path.exists():
        try:
            with open(mushkil_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter empty lines and very short lines
            filtered = [
                line.strip() + '\n' 
                for line in lines 
                if line.strip() and len(line.strip()) > 15
            ]
            
            all_lines.extend(filtered)
            print(f"  ✓ مشكل الآثار.txt: {len(filtered)} lines")
        
        except Exception as e:
            print(f"  ✗ Error reading مشكل الآثار.txt: {e}")
    else:
        print(f"  ⚠ مشكل الآثار.txt not found")
    
    # 3. Read 25k lines from train.txt
    print(f"\n[3] Reading 25,000 lines from data/train.txt...")
    train_path = Path("data/train.txt")
    
    if train_path.exists():
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()
            
            # Take first 25k lines
            train_subset = train_lines[:25000]
            all_lines.extend(train_subset)
            print(f"  ✓ Added {len(train_subset)} lines from train.txt")
        
        except Exception as e:
            print(f"  ✗ Error reading train.txt: {e}")
    else:
        print(f"  ⚠ train.txt not found at {train_path}")
    
    # 4. Write output
    output_path = Path("data/train_modern_mix.txt")
    print(f"\n[4] Writing combined dataset to {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        
        print(f"  ✓ Successfully wrote {len(all_lines)} lines")
    
    except Exception as e:
        print(f"  ✗ Error writing output: {e}")
        return
    
    # 5. Statistics
    print("\n" + "=" * 80)
    print("Dataset Creation Complete")
    print("=" * 80)
    print(f"Modern books (كتب حديثة): {modern_count:,} lines")
    print(f"Classical hadith (مشكل الآثار): Included")
    print(f"Train.txt subset: 25,000 lines")
    print(f"Total lines: {len(all_lines):,}")
    print(f"\nOutput: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    concatenate_datasets()
