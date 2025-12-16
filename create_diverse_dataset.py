"""
Create a diverse training dataset by concatenating mixed-genre Arabic texts.
Strategy: Balance modern news, historical narratives, literature, and some fiqh.
"""

import os
from pathlib import Path

# Define source files with their genres
SOURCE_FILES = {
    # Modern Standard Arabic (30-35% target)
    "msa/aljazeera/aljazeera.txt": "modern_news",
    "msa/aljazeera/aljazeera-2016-12-29.b.txt": "modern_news",
    "msa/منوع/أرق.txt": "modern_essay",
    "msa/منوع/الجراح في ايام الحب.txt": "modern_essay",
    "msa/منوع/الكثير في التفاصيل.txt": "modern_essay",
    "msa/منوع/إملاء رابع المنهج الجديد ف1.txt": "modern_educational",
    "msa/منوع/المعالم الجغرافية الواردة في السيرة النبوية.htm.txt": "modern_reference",
    "msa/منوع/اليهود في مملكتي قشتالة وأراجون.txt": "modern_history",
    
    # Historical narratives (20-25% target)
    "تاريخ الإسلام 1 و 2 المشكول.txt": "historical",
    "سيرة ابن هشام.txt": "historical",
    "مغازي الواقدي.txt": "historical",
    
    # Classical literature (15-20% target)
    "أدب الدنيا والدين.txt": "literature",
    "الآداب الشرعية.txt": "literature",
    
    # Keep some fiqh for domain knowledge (25-30% target)
    # Will sample from existing train.txt
}

def get_line_count(filepath):
    """Count lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error counting lines in {filepath}: {e}")
        return 0

def concatenate_files(texts_dir, output_file, target_lines=50000):
    """
    Concatenate diverse texts to create balanced training dataset.
    
    Args:
        texts_dir: Path to texts.txt directory
        output_file: Output path for new training file
        target_lines: Target number of lines (~50K to match original)
    """
    
    print("=" * 80)
    print("Creating Diverse Training Dataset")
    print("=" * 80)
    
    all_lines = []
    
    # 1. Process diverse source files
    print("\n[1] Processing diverse source files...")
    for relative_path, genre in SOURCE_FILES.items():
        filepath = os.path.join(texts_dir, relative_path)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ SKIP: {relative_path} (not found)")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter out empty lines and metadata
            filtered_lines = [
                line.strip() + '\n' 
                for line in lines 
                if line.strip() and not line.startswith('http') and len(line.strip()) > 20
            ]
            
            all_lines.extend(filtered_lines)
            print(f"  ✓ {relative_path}: {len(filtered_lines)} lines ({genre})")
        
        except Exception as e:
            print(f"  ✗ ERROR reading {relative_path}: {e}")
    
    # 2. Add subset of original fiqh data for domain knowledge
    print("\n[2] Sampling fiqh subset from original train.txt...")
    train_txt_path = os.path.join(os.path.dirname(texts_dir), "data", "train.txt")
    
    if os.path.exists(train_txt_path):
        try:
            with open(train_txt_path, 'r', encoding='utf-8') as f:
                fiqh_lines = f.readlines()
            
            # Sample ~25% of original fiqh (12,500 lines)
            fiqh_sample_size = min(12500, len(fiqh_lines) // 4)
            
            # Take stratified sample (beginning, middle, end)
            step = len(fiqh_lines) // fiqh_sample_size
            fiqh_sample = [fiqh_lines[i] for i in range(0, len(fiqh_lines), step)][:fiqh_sample_size]
            
            all_lines.extend(fiqh_sample)
            print(f"  ✓ Sampled {len(fiqh_sample)} fiqh lines from train.txt")
        
        except Exception as e:
            print(f"  ✗ ERROR reading train.txt: {e}")
    else:
        print(f"  ⚠ train.txt not found at {train_txt_path}")
    
    # 3. Shuffle for diversity (optional - comment out to preserve order)
    # import random
    # random.seed(42)
    # random.shuffle(all_lines)
    
    # 4. Write output
    print(f"\n[3] Writing output to {output_file}...")
    total_lines = len(all_lines)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        
        print(f"  ✓ Successfully wrote {total_lines} lines")
    
    except Exception as e:
        print(f"  ✗ ERROR writing output: {e}")
        return
    
    # 5. Statistics
    print("\n" + "=" * 80)
    print("Dataset Creation Complete")
    print("=" * 80)
    print(f"Total lines: {total_lines}")
    print(f"Output file: {output_file}")
    print(f"\n📊 Genre Distribution Estimate:")
    print(f"  Modern content: ~35-40% (news, essays, sermons)")
    print(f"  Historical narratives: ~20-25%")
    print(f"  Classical literature: ~10-15%")
    print(f"  Fiqh (domain knowledge): ~25-30%")
    print("\n✅ Ready to train with diverse data!")
    print("   Command: python src/train.py --model arabert_char_bilstm_crf \\")
    print(f"            --train_data {output_file} --val_data data/val.txt")
    print("=" * 80)

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent
    texts_dir = project_root / "texts.txt"
    output_file = project_root / "data" / "train_diverse.txt"
    
    # Create dataset
    concatenate_files(str(texts_dir), str(output_file), target_lines=50000)
