# Arabic Diacritization System - AI Coding Agent Guide

## Project Overview

Arabic diacritization system using BiLSTM-CRF and contextual embeddings (AraBERT). Predicts 15 diacritic classes for each character in Arabic text sequences.

## Core Architecture

### Data Flow Pattern

```
Raw Text → tokenize.py → Character-Diacritic Pairs → encode_labels.py → Integer IDs (0-14) → Model → Predictions
```

**Critical alignment**: Characters and diacritic labels maintain strict 1-to-1 correspondence. Spaces are skipped during tokenization but word boundaries are implicit.

### Dual Embedding Modes

The system supports two distinct embedding strategies controlled by `use_contextual` flag:

1. **Character Embeddings** (`use_contextual=False`):

   - Simple learned embeddings (100-dim)
   - Fast training on CPU
   - vocab.py builds char2id mapping with `<PAD>=0`, `<UNK>=1`
   - Uses `TensorDataset` with pre-encoded sequences

2. **Contextual Embeddings** (`use_contextual=True`):
   - AraBERT generates 768-dim per-character embeddings
   - Requires GPU, uses custom `ContextualDataset` class
   - Embeddings computed on-the-fly in `__getitem__` via `embedder.embed_line_chars()`
   - Must use `collate_contextual_batch` for dynamic padding

**Key Convention**: When switching modes, update BOTH `embedding_dim` (100→768) AND `use_contextual` (False→True) in config.py

## Critical Configuration (src/config.py)

All models use centralized config dictionaries. Example pattern:

```python
BILSTM_CRF_CONFIG = {
    "embedding_dim": 100,        # 768 for AraBERT
    "hidden_dim": 256,           # BiLSTM effective (128 per direction)
    "tagset_size": 15,           # Fixed diacritic classes
    "use_contextual": False,     # True for AraBERT
    "batch_size": 32,            # 1 for contextual (memory)
    "num_epochs": 50,
    "patience": 7,               # Early stopping
}
```

**Convention**: Config changes require retraining. Never modify configs mid-training.

## Model Implementations

### BiLSTM-CRF (Baseline Model)

- **File**: [src/models/bilstm_crf.py](src/models/bilstm_crf.py)
- **CRF Library**: TorchCRF (requires tensor transposition: batch_first → seq_first)
- **Forward behavior**: Returns loss (training) or predictions list (inference) based on `tags` parameter
- **Transposition pattern** (critical for CRF):
  ```python
  emissions_transposed = emissions.transpose(0, 1)  # (batch, seq, tags) → (seq, batch, tags)
  tags_transposed = tags.transpose(0, 1)
  loss = -self.crf(emissions_transposed, tags_transposed, mask=mask).sum()
  ```

### AraBERT + Character Fusion BiLSTM-CRF (SOTA Model)

- **File**: [src/models/arabert_char_bilstm_crf.py](src/models/arabert_char_bilstm_crf.py)
- **Dual Feature Architecture**: Combines AraBERT contextual embeddings (semantic) with character embeddings (morphological)
- **Forward signature**: `forward(arabert_emb, char_ids, tags=None, mask=None)`
- **Key components**:
  - AraBERT projection (768→384) with LayerNorm + Dropout
  - Character embedding layer (100-dim)
  - Feature concatenation (484-dim total)
  - 2-layer BiLSTM (deeper than baseline)
  - CRF decoding
- **Training**: Requires dual inputs unpacked from 4-tuple batches: `(embedding, char_ids, labels, mask)`

### Hierarchical BiLSTM

- **File**: [src/models/hierarchical_bilstm.py](src/models/hierarchical_bilstm.py)
- Uses character BiLSTM + 1D convolution (kernel_size=3) for word features
- No CRF layer, uses standard cross-entropy loss

## Training & Testing Workflow

### Standard Training Commands

```bash
# CPU with character embeddings (4-8 hours) - BASELINE
python src/train.py --model bilstm_crf --train_data data/train.txt --val_data data/val.txt

# GPU with AraBERT - BASELINE (97.99% accuracy)
python src/train.py --model bilstm_crf --train_data data/train.txt --val_data data/val.txt

# GPU with AraBERT + Character Fusion - SOTA (expected 98.2-98.5%)
python src/train.py --model arabert_char_bilstm_crf --train_data data/train.txt --val_data data/val.txt
```

### Testing

```bash
# Test baseline model
python src/test.py --model bilstm_crf --model_path models/best_bilstm_crf.pth --test_data data/test.txt

# Test SOTA fusion model
python src/test.py --model arabert_char_bilstm_crf --model_path models/best_arabert_char_bilstm_crf.pth --test_data data/test.txt
```

**Convention**: Always specify `--model` matching the architecture used during training.

## Preprocessing Pipeline

### Tokenization (src/preprocessing/tokenize.py)

- **Input**: Diacritized Arabic text
- **Output**: Parallel lists (X: base chars, Y: diacritic strings)
- **Diacritic Canonicalization**: Shadda (ّ) always comes first, then other diacritics
- **Unicode Normalization**: NFKC → NFC to handle compatibility forms

### Label Encoding (src/preprocessing/encode_labels.py)

- Uses `utils/diacritic2id.pickle` for mapping
- Empty string (`''`) = 0 (no diacritic)
- Returns list of integer sequences matching character positions

### Padding (src/preprocessing/pad_sequences.py)

- Default `max_seq_length=256` (configurable)
- Padding value: 0 (matches `<PAD>` and empty diacritic)
- Truncation: Keeps leftmost characters

## Contextual Embeddings (src/features/contextual_embeddings.py)

### ContextualEmbedder Class

- **Model**: `aubmindlab/bert-base-arabertv02`
- **Key Method**: `embed_line_chars(line)` returns (T, 768) per-character embeddings
- **Input Preprocessing**: Strips diacritics before feeding to BERT
- **Caching**: Disabled by default (`cache_dir=None`) to save disk space
- **Alignment**: Word-level BERT embeddings expanded to characters within each word

**Important**: Contextual embeddings require matching `char_ids` for morphology fusion in some models. See ContextualDataset implementation for dual-output pattern.

## Data Files & Vocabulary

### Vocabulary Building

- **Character Vocab**: [utils/vocab.py](utils/vocab.py) - CharVocab class builds incrementally
- **Diacritic Mapping**: `utils/diacritic2id.pickle` - Fixed 15-class mapping
- **Word Vocab**: [utils/word_vocab.py](utils/word_vocab.py) - For hierarchical models

### Data Structure

```
data/
  train.txt          # 50,001 lines diacritized Arabic
  val.txt            # Validation split
  processed/
    contextual_cache/ # AraBERT embedding cache (optional)
models/
  best_bilstm_crf.pth              # Character embeddings model
  best_hierarchical_bilstm.pth     # Hierarchical model
```

## Custom Dataset Pattern (Contextual Mode)

```python
class ContextualDataset(Dataset):
    def __getitem__(self, idx):
        line = self.lines[idx]
        emb = self.embedder.embed_line_chars(line)  # (T, 768)
        char_ids = self.vocab.encode(chars)         # (T,) for morphology
        return {
            "embedding": torch.tensor(emb),
            "char_ids": torch.tensor(char_ids),
            "label": torch.tensor(y_seq),
            "mask": torch.ones(T, dtype=torch.bool)
        }

def collate_contextual_batch(batch):
    # Pad embeddings, char_ids, labels, masks to max_len in batch
    # Uses torch.nn.functional.pad with appropriate dimensions
```

**Convention**: Always use `batch_size=1` for contextual mode due to memory constraints.

## Evaluation Metrics

### DER (Diacritic Error Rate)

- Primary metric: `errors / total_characters` (excluding padding)
- Expected range: 5-15% (lower is better)
- Calculated per character, not per word

### Accuracy

- `correct_predictions / total_characters` = (1 - DER)
- Expected range: 85-95%

**Convention**: Spaces are excluded from all metrics. Only count actual base characters.

## Common Pitfalls

1. **CRF Tensor Shape Mismatch**: Always transpose from batch_first to seq_first before CRF forward
2. **Config Sync Issues**: Changing `use_contextual` requires updating `embedding_dim` (100↔768) and `batch_size` (32↔1)
3. **Vocabulary Mismatches**: Character vocab must be built from SAME data split used for training
4. **Padding Semantics**: Padding value 0 represents both `<PAD>` token and empty diacritic class
5. **Memory Issues with AraBERT**: Use `batch_size=1` and disable caching if running out of memory

## Dependencies

See [requirements.txt](requirements.txt) for versions:

- torch>=2.0.0
- transformers>=4.30.0 (for AraBERT)
- TorchCRF>=1.1.0 (note: requires tensor transposition)
- numpy, scikit-learn, tqdm

## Project-Specific Conventions

- **Seed**: All scripts use `seed=42` for reproducibility via `set_seed()` function
- **Gradient Clipping**: `max_norm=5.0` for all models
- **Optimizer**: Adam with `weight_decay=1e-5`
- **LR Scheduling**: StepLR (step_size=10, gamma=0.5)
- **Early Stopping**: Monitors validation DER with configurable patience (default: 7 epochs)
- **Model Checkpointing**: Saves complete state including optimizer/scheduler for resumption

## Key Architecture Documents

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Detailed data flow diagrams
- [HIERARCHICAL_MODEL_ARCHITECTURE.md](HIERARCHICAL_MODEL_ARCHITECTURE.md) - Component specifications
- [VERIFIED_FLOW_SUMMARY.md](VERIFIED_FLOW_SUMMARY.md) - Training commands and configurations
- [ARABERT_CHAR_FUSION_IMPLEMENTATION.md](ARABERT_CHAR_FUSION_IMPLEMENTATION.md) - SOTA fusion model implementation guide
