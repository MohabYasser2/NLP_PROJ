# Arabic Diacritization Project - AI Assistant Guide

## Project Overview
This is an **Arabic text diacritization** system using deep learning. The task is sequence labeling: predict which diacritic marks (ً ٌ ٍ َ ُ ِ ْ ّ) should be added to each Arabic base character. The system uses **15 diacritic classes** including combinations like "ّ َ" (shadda+fatha) and "" (no diacritic).

**Key concept**: The model predicts one diacritic label per base character, NOT per Unicode character. Preprocessing tokenizes text into (base_chars, diacritics) aligned sequences.

## Architecture & Data Flow

### Core Pipeline
1. **Tokenization** ([src/preprocessing/tokenize.py](src/preprocessing/tokenize.py)): Splits Arabic text into base characters (letters/digits) and their diacritics
2. **Encoding** ([src/preprocessing/encode_labels.py](src/preprocessing/encode_labels.py)): Maps diacritic strings to IDs using `utils/diacritic2id.pickle` (single source of truth)
3. **Feature extraction** ([src/features/](src/features/)): 
   - Character embeddings (100-dim trainable)
   - AraBERT contextual embeddings (768-dim, `aubmindlab/bert-base-arabertv02`)
   - N-gram features (bigrams)
4. **Models** ([src/models/](src/models/)): BiLSTM-CRF variants with/without contextual embeddings
5. **Inference**: Decode CRF predictions back to diacritized text

### Model Variants (Best → Simple)
- **`arabert_char_bilstm_crf`**: SOTA - Dual fusion (AraBERT 768-dim + Character 100-dim) → 2-layer BiLSTM → CRF
- **`arabert_bilstm_crf`**: AraBERT only → BiLSTM → CRF
- **`char_bilstm_classifier`**: Character embeddings → BiLSTM → Softmax (no CRF)
- **`charngram_bilstm_classifier`**: Character + Bigram → BiLSTM → Softmax
- **`bilstm_crf`**: Simple character-only BiLSTM-CRF baseline

All models output **15 classes** per character (NUM_DIACRITIC_CLASSES in [src/config.py](src/config.py)).

## Critical Patterns & Conventions

### 1. Model Input Handling (Forward Pass Signatures)
Models have **different input requirements**. Always check model type before feeding data:

```python
# CRF models (bilstm_crf, arabert_bilstm_crf)
loss = model(X_batch, tags=y_batch, mask=mask_batch)  # Training
predictions = model(X_batch, mask=mask_batch)          # Inference

# Fusion model (arabert_char_bilstm_crf) - DUAL INPUTS
loss = model(arabert_emb, char_ids, tags=y_batch, mask=mask_batch)
predictions = model(arabert_emb, char_ids, mask=mask_batch)

# Classifier models (char_bilstm_classifier, charngram_bilstm_classifier)
logits, loss = model(X_batch, tags=y_batch, mask=mask_batch)  # Training
logits, preds = model(X_batch, mask=mask_batch)               # Inference
```

**Train/test scripts** ([src/train.py](src/train.py), [src/test.py](src/test.py)) use flags to detect model type:
```python
is_fusion_model = model_name.lower() == "arabert_char_bilstm_crf"
is_simple_classifier = model_name.lower() == "char_bilstm_classifier"
```

### 2. Dataset Creation (Contextual vs Non-Contextual)
- **Non-contextual** (character embeddings): Use `TensorDataset` with pre-encoded sequences
- **Contextual** (AraBERT): Use `ContextualDataset` that computes embeddings on-the-fly to save memory
  - Batch size must be **1** for contextual models (see DATA_CONFIG in config.py)
  - Use `collate_contextual_batch` collate function for padding

### 3. Configuration System ([src/config.py](src/config.py))
- Each model has a dedicated config dict (e.g., `ARABERT_CHAR_BILSTM_CRF_CONFIG`)
- **Vocab size is set dynamically** - don't hardcode it:
  ```python
  config = get_model_config(model_name)
  config = update_vocab_size(config.copy(), len(vocab.char2id))
  ```
- Key flag: `use_contextual` determines if AraBERT embeddings are used

### 4. Evaluation Metrics
- **DER (Diacritic Error Rate)**: Primary metric - % of characters with wrong diacritics
- **Accuracy**: Character-level accuracy (excludes padding and empty diacritics "")
- Both metrics **exclude padding tokens** using the mask tensor

### 5. Data Files
- Training data: [data/train.txt](data/train.txt), [data/val.txt](data/val.txt) - already tokenized/normalized
- Test data: CSV format with `id,text` (no diacritics) → predict → CSV with `id,text_diacritized`
- Raw corpus: [texts.txt/](texts.txt/) contains classical Arabic books for data augmentation

## Development Workflows

### Training a Model
```powershell
python -m src.train --model arabert_char_bilstm_crf --train data/train.txt --val data/val.txt
```
- Models auto-save to `models/best_{model_name}.pth` when validation DER improves
- Checkpoints include: model weights, optimizer state, config, vocabulary

### Testing/Inference
```powershell
python -m src.test --model arabert_char_bilstm_crf --checkpoint models/best_arabert_char_bilstm_crf.pth --test data/test.csv --output predictions.csv
```

### Data Normalization
```powershell
python normalize_dataset.py  # Normalizes data to 15-diacritic system
```
Filters invalid diacritics and ensures consistency with `utils/diacritic2id.pickle`.

### Adding a New Model
1. Create model class in [src/models/](src/models/) inheriting `nn.Module`
2. Add config dict to [src/config.py](src/config.py) and update `get_model_config()`
3. Add model initialization branch in `get_model()` function ([src/train.py](src/train.py) lines 200-250)
4. Add evaluation branch in `evaluate_model()` if input signature differs

## Gotchas & Important Notes

1. **Diacritic Mapping**: ALWAYS load from `utils/diacritic2id.pickle` - never hardcode diacritic IDs
2. **Vocab Leakage**: Build vocabulary ONLY from training data (see [src/train.py](src/train.py) line 430)
3. **Batch Size**: Set to 1 for contextual models, 32 for non-contextual (memory constraints)
4. **Sequence Alignment**: Tokenization produces (base_chars, diacritics) where len(base_chars) == len(diacritics)
5. **CRF Models**: Return lists of predictions (not tensors) - handle accordingly in evaluation
6. **AraBERT Cache**: Disabled on Kaggle (`cache_dir=None`) to avoid disk space issues
7. **Gradient Clipping**: All models use gradient clipping (config["gradient_clip"]) to prevent exploding gradients
8. **Early Stopping**: Patience counter resets when DER improves (not accuracy)

## File Hierarchy Quick Reference
```
src/
  config.py              # All hyperparameters (single source of truth)
  train.py               # Training script with model-specific branching
  test.py                # Inference script for CSV predictions
  models/                # Model architectures (inheritance: nn.Module)
  preprocessing/         # Tokenization & encoding (1-to-1 char alignment)
  features/              # Feature extractors (AraBERT, n-grams)
utils/
  vocab.py               # CharVocab class (char2id, id2char)
  diacritic2id.pickle    # 15-class label mapping (IMMUTABLE)
data/
  train.txt, val.txt     # Tokenized training data (1 line = 1 sample)
models/                  # Saved checkpoints (.pth files)
```

## When Working With This Codebase
- **Read model forward signatures** before modifying train/test loops
- **Check config.py first** when adjusting hyperparameters
- **Use grep_search** to find how existing models handle specific inputs
- **Test with small max_samples** (e.g., 100) before full training runs
- **Preserve mask handling** in all sequence operations (critical for CRF)
