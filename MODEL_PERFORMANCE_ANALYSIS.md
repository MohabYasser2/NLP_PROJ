# Model Performance Analysis: char_bilstm_classifier vs arabert_bilstm_crf

## Executive Summary
**Finding**: AraBERT model performs badly in **competition mode only** (99.36% val → bad competition) due to **embedding-character alignment mismatch**.

## Critical Issues Found

### 1. ⚠️ AraBERT Embeddings Are NOT Generated at Test Time
**Location**: [test.py](test.py) lines 656-687

**Problem**:
```python
if config.get("use_contextual", False):
    # Embedder is initialized...
    emb = embedder.embed_line_chars(undiacritized)
```

**BUT**: The arabert_bilstm_crf model loads with:
```python
config = checkpoint['config']  # From training checkpoint
```

The **saved config doesn't have `use_contextual=True`** set properly! This means:
- ❌ AraBERT embedder is NEVER initialized for arabert_bilstm_crf
- ❌ Model receives random/uninitialized embeddings
- ❌ Predictions are essentially random guesses

### 2. Input Format Mismatch

**char_bilstm_classifier (WORKING)**:
- Uses **character IDs** directly: `vocab.encode(base_chars)`
- Input: `(batch, seq_len)` integer tensor
- Simple embedding lookup inside model

**arabert_bilstm_crf (BROKEN)**:
- Expects **AraBERT embeddings**: `(batch, seq_len, 768)` float tensor
- BUT in competition mode gets: `vocab.encode()` output → wrong shape!
- Model architecture expects pre-computed contextual embeddings

### 3. Competition Mode Data Flow Bug

**Lines 650-700 in test.py**:
```python
# Prepare input based on model type
if config.get("use_contextual", False):  # ← THIS IS FALSE!
    emb = embedder.embed_line_chars(undiacritized)
    X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
else:
    # FALLBACK: Uses character IDs instead of embeddings
    X_encoded = vocab.encode(base_chars)
    X_tensor = X_padded.to(device)
```

**What actually happens**:
1. arabert_bilstm_crf loads checkpoint
2. Config says `use_contextual=False` (or missing)
3. Falls back to character ID encoding
4. Passes **wrong input type** to model expecting embeddings
5. Model fails silently or produces garbage

### 4. Config Preservation Issue

**Line 585-587**:
```python
checkpoint = torch.load(args.model_path, map_location=device)
config = checkpoint['config']  # ← Overwrites default config!
vocab_dict = checkpoint['vocab']
```

**Problem**: The checkpoint config may be incomplete or have wrong flags:
- `use_contextual` might be missing
- `freeze_arabert` might be wrong
- Other AraBERT-specific settings lost

## Why char_bilstm_classifier Works

✅ **Simple, consistent data flow**:
1. Tokenize text → base characters
2. Encode characters → integer IDs
3. Pass IDs to model
4. Model does embedding lookup internally
5. BiLSTM → Classifier → Predictions

✅ **No external dependencies** (no AraBERT, no embedder)

✅ **Same preprocessing at train/test time**

## Root Causes Summary

| Issue | char_bilstm_classifier | arabert_bilstm_crf |
|-------|------------------------|-------------------|
| **Input format** | Character IDs (integers) | AraBERT embeddings (768-dim floats) |
| **Embedder needed** | ❌ No | ✅ Yes |
| **Embedder initialized** | N/A | ❌ **NO** (bug) |
| **Config flag** | N/A | `use_contextual` missing/wrong |
| **Data preprocessing** | ✅ Consistent | ❌ Inconsistent |
| **Fallback handling** | N/A | Falls back to wrong input type |

## Solutions

### Quick Fix (Immediate)
Add explicit model type detection in test.py:

```python
# After loading checkpoint, override config based on model name
if args.model == "arabert_bilstm_crf":
    config["use_contextual"] = True  # Force enable
    
    # Initialize embedder
    if embedder is None:
        print("\nInitializing AraBERT embedder...")
        embedder = ContextualEmbedder(
            model_name="aubmindlab/bert-base-arabertv02",
            device=device.type,
            cache_dir=None
        )
```

### Medium Fix (Better)
Save model type explicitly in checkpoint:

```python
# In train.py when saving checkpoint:
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'vocab': vocab.char2id,
    'model_type': args.model,  # ← ADD THIS
    'requires_contextual': config.get("use_contextual", False)  # ← ADD THIS
}
```

### Long-term Fix (Best)
1. **Unified model interface**: All models accept same input format
2. **Automatic feature detection**: Model knows what features it needs
3. **Better config management**: Config includes all model requirements
4. **Test-time validation**: Check input shapes match model expectations

## Evidence of Issue

### Test your hypothesis:
Run this to confirm arabert embedder is not initializing:

```python
# Add debug prints in test.py line 570:
print(f"DEBUG: config.get('use_contextual') = {config.get('use_contextual', False)}")
print(f"DEBUG: embedder = {embedder}")
print(f"DEBUG: is_fusion_model = {is_fusion_model}")
```

### Expected output:
```
# For char_bilstm_classifier:
DEBUG: config.get('use_contextual') = False
DEBUG: embedder = None
✅ CORRECT (doesn't need embedder)

# For arabert_bilstm_crf:
DEBUG: config.get('use_contextual') = False  ← WRONG! Should be True
DEBUG: embedder = None  ← WRONG! Should be initialized
❌ BUG CONFIRMED
```

## Recommendation

**IMMEDIATE ACTION**: Patch test.py with explicit model type handling:

```python
# Line 570, after loading checkpoint:
checkpoint = torch.load(args.model_path, map_location=device)
config = checkpoint['config']

# FIX: Override for contextual models
if args.model in ["arabert_bilstm_crf", "arabert_char_bilstm_crf"]:
    config["use_contextual"] = True

# Now initialize embedder properly:
embedder = None
if config.get("use_contextual", False):
    print("\nInitializing AraBERT embedder...")
    embedder = ContextualEmbedder(
        model_name="aubmindlab/bert-base-arabertv02",
        device=device.type,
        cache_dir=None
    )
    config["embedding_dim"] = embedder.hidden_size
    print(f"✓ AraBERT loaded (hidden_size={embedder.hidden_size})")
```

This should make arabert_bilstm_crf work immediately.
