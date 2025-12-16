# Arabic Diacritization Project - AI Assistant Guide

## Project Overview

This is an **Arabic text diacritization** system using deep learning. The task is sequence labeling: predict which diacritic marks (ً ٌ ٍ َ ُ ِ ْ ّ) should be added to each Arabic base character. The system uses **15 diacritic classes** including combinations like "ّ َ" (shadda+fatha) and "" (no diacritic).

**Key concept**: The model predicts one diacritic label per base character, NOT per Unicode character. Preprocessing tokenizes text into (base_chars, diacritics) aligned sequences.

## Architecture & Data Flow

### Core Pipeline

1. **Tokenization** ([src/preprocessing/tokenize.py](src/preprocessing/tokenize.py)): Splits Arabic text into base characters (letters/digits) and their diacritics using `tokenize_line()` → returns `(base_chars, diacritics)` aligned lists
2. **Encoding** ([src/preprocessing/encode_labels.py](src/preprocessing/encode_labels.py)): Maps diacritic strings to IDs using `utils/diacritic2id.pickle` (single source of truth - NEVER modify)
3. **Feature extraction** ([src/features/](src/features/)):
   - Character embeddings (100-128 dim trainable) - embedded inside model
   - AraBERT contextual embeddings (768-dim, `aubmindlab/bert-base-arabertv02`) via [src/features/contextual_embeddings.py](src/features/contextual_embeddings.py)
   - N-gram features (bigrams) via [src/features/ngram_features.py](src/features/ngram_features.py)
4. **Models** ([src/models/](src/models/)): BiLSTM-CRF variants with/without contextual embeddings
5. **Inference**: Decode CRF predictions back to diacritized text

### Model Variants (Best → Simple)

- **`arabert_char_bilstm_crf`**: SOTA - Dual fusion (AraBERT 768-dim + Character 100-dim) → 2-layer BiLSTM → CRF
  - **REQUIRES:** AraBERT embedder initialization + dual input tensors (embeddings + char_ids)
- **`arabert_bilstm_crf`**: AraBERT only → BiLSTM → CRF (embeddings-only input)
- **`char_bilstm_classifier`**: Character embeddings → BiLSTM → Softmax (no CRF, simplest)
- **`charngram_bilstm_classifier`**: Character + Bigram → BiLSTM → Softmax (dual vocabulary)
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

**Detection pattern** used in [src/test.py](src/test.py) (model introspection):

```python
is_fusion_model = hasattr(model, 'char_embedding') and hasattr(model, 'arabert_projection')
is_simple_classifier = hasattr(model, 'char_embedding') and not hasattr(model, 'arabert_projection')
is_ngram_classifier = hasattr(model, 'ngram_embedding')
```

**CRITICAL:** For contextual models, check `config.get("use_contextual", False)` - if True, embedder MUST be initialized.

### 2. Dataset Creation (Contextual vs Non-Contextual)

- **Non-contextual** (character embeddings): Use `TensorDataset` with pre-encoded sequences
- **Contextual** (AraBERT): Use `ContextualDataset` that computes embeddings on-the-fly to save memory
  - Embedder requires initialization: `ContextualEmbedder(model_name="aubmindlab/bert-base-arabertv02", device=device.type, cache_dir=None)`
  - Batch size must be **1** for contextual models during training (memory constraints)
  - Use `collate_contextual_batch` collate function for padding

### 3. Configuration System ([src/config.py](src/config.py))

- Each model has a dedicated config dict (e.g., `ARABERT_CHAR_BILSTM_CRF_CONFIG`)
- **Vocab size is set dynamically** - don't hardcode it:
  ```python
  config = get_model_config(model_name)
  config = update_vocab_size(config.copy(), len(vocab.char2id))
  ```
- Key flag: `use_contextual` determines if AraBERT embeddings are used
- **Config persistence:** When saving checkpoints, config dict is saved alongside model weights - must be reloaded correctly during inference

### 4. Evaluation Metrics

- **DER (Diacritic Error Rate)**: Primary metric - % of characters with wrong diacritics
- **Accuracy**: Character-level accuracy (excludes padding and empty diacritics "")
- Both metrics **exclude padding tokens** using the mask tensor

### 5. Data Files

- Training data: [data/train.txt](data/train.txt), [data/val.txt](data/val.txt) - already tokenized/normalized
- Test data: Text files with undiacritized Arabic text (one line per sample)
- Competition format: CSV with `ID,label` columns where ID is sequential character index across all lines
- Raw corpus: [texts.txt/](texts.txt/) contains classical Arabic books for data augmentation

## Development Workflows

### Training a Model

```powershell
python -m src.train --model arabert_char_bilstm_crf --train data/train.txt --val data/val.txt
```

- Models auto-save to `models/best_{model_name}.pth` when validation DER improves
- Checkpoints include: model weights, optimizer state, config, vocabulary

### Testing/Inference (Evaluation Mode)

```powershell
# Evaluate on validation data
python -m src.test --model arabert_char_bilstm_crf --model_path "models/best_arabert_char_bilstm_crf.pth" --test_data data/val.txt

# Test with limited samples for quick debugging
python -m src.test --model char_bilstm_classifier --model_path "models/best_char_bilstm_classifier.pth" --test_data data/val.txt --max_samples 100
```

### Competition Mode (CSV Output)

Generate predictions in competition format (ID, label) for submission:

```powershell
# Standard competition output (ID, label)
python -m src.test --model arabert_char_bilstm_crf --model_path "models/best_arabert_char_bilstm_crf.pth" --competition --competition_input test_no_diacritics.txt --competition_output submission.csv

# Debug mode with extra columns (line_number, letter, case_ending, predicted_diacritic)
python -m src.test --model char_bilstm_classifier --model_path "models/best_char_bilstm_classifier.pth" --competition --competition_input test_no_diacritics.txt --competition_output debug.csv --competition_debug
```

**Competition mode specifics:**

- Input: Plain text file with undiacritized Arabic (one line per sample)
- Output: CSV with sequential character IDs and predicted diacritic labels
- Uses same tokenization as training (`tokenize_line()`) to ensure consistency
- **Exit behavior:** Script exits after generating CSV (doesn't proceed to evaluation)

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

## Debugging & Troubleshooting

### Common Issues

**AraBERT Model Fails in Competition Mode**

- **Symptom**: Model has 99%+ validation DER but terrible competition predictions
- **Root Cause**: `use_contextual` flag not set in checkpoint config → embedder not initialized → wrong input type
- **Fix**: Verify checkpoint contains correct config flags, or override in test.py based on model name
- **Reference**: See [MODEL_PERFORMANCE_ANALYSIS.md](MODEL_PERFORMANCE_ANALYSIS.md) for complete diagnosis

**Input Shape Mismatches**

- **Symptom**: RuntimeError about tensor shapes during forward pass
- **Diagnosis**: Use model introspection to detect model type:
  ```python
  is_fusion_model = hasattr(model, 'char_embedding') and hasattr(model, 'arabert_projection')
  is_ngram_classifier = hasattr(model, 'ngram_embedding')
  ```
- **Fix**: Match input format to model architecture (see "Model Input Handling" section)

**Memory Issues During Training**

- **Symptom**: CUDA OOM or system memory exhausted
- **Fix**: Reduce batch size to 1 for contextual models, disable AraBERT cache (`cache_dir=None`)
- **Alternative**: Use gradient accumulation or mixed precision training

**CRF Prediction Format**

- **Symptom**: Predictions are nested lists `[[1], [2], [3]]` instead of flat `[1, 2, 3]`
- **Fix**: Check CRF return format and flatten if needed (see [src/test.py](src/test.py) line 680-695)

### Debugging Commands

```powershell
# Test with limited samples for fast iteration
python -m src.test --model MODEL_NAME --model_path PATH --test_data data/val.txt --max_samples 100

# Debug competition output with extra columns
python -m src.test --model MODEL_NAME --model_path PATH --competition --competition_input INPUT.txt --competition_output OUTPUT.csv --competition_debug

# Check config persistence in checkpoint
python -c "import torch; ckpt = torch.load('models/best_MODEL.pth'); print(ckpt['config'])"
```

## Gotchas & Important Notes

1. **Diacritic Mapping**: ALWAYS load from `utils/diacritic2id.pickle` - never hardcode diacritic IDs
2. **Vocab Leakage**: Build vocabulary ONLY from training data (see [src/train.py](src/train.py) line 430)
3. **Batch Size**: Set to 1 for contextual models, 32 for non-contextual (memory constraints)
4. **Sequence Alignment**: Tokenization produces (base_chars, diacritics) where len(base_chars) == len(diacritics)
5. **CRF Models**: Return lists of predictions (not tensors) - handle accordingly in evaluation
6. **AraBERT Cache**: Disabled on Kaggle (`cache_dir=None`) to avoid disk space issues
7. **Gradient Clipping**: All models use gradient clipping (config["gradient_clip"]) to prevent exploding gradients
8. **Early Stopping**: Patience counter resets when DER improves (not accuracy)
9. **AraBERT Embedder Bug**: For contextual models, if `config.get("use_contextual")` is False or missing from checkpoint, embedder won't initialize → model receives wrong input type → predictions fail. See [MODEL_PERFORMANCE_ANALYSIS.md](MODEL_PERFORMANCE_ANALYSIS.md) for detailed diagnosis.
10. **Competition vs Evaluation Mode**: `--competition` flag changes behavior - script exits after CSV generation instead of proceeding to evaluation metrics

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
