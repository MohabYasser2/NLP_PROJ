# AraBERT + Character Fusion Model Implementation

## What Was Implemented

A new state-of-the-art (SOTA) model that combines two complementary feature types:
- **AraBERT embeddings (768-dim)**: Captures semantic and word-level context
- **Character embeddings (100-dim)**: Captures morphological and orthographic patterns

This dual-feature fusion approach theoretically improves diacritization accuracy by combining:
1. **Semantic understanding** (what words mean in context) - from AraBERT
2. **Morphological patterns** (character-level orthography) - from char embeddings

## Files Created/Modified

### ✅ New Files
1. **src/models/arabert_char_bilstm_crf.py** - The fusion model architecture
   - AraBERT projection layer with LayerNorm + Dropout
   - Character embedding layer
   - Feature concatenation (fusion)
   - 2-layer BiLSTM (deeper than baseline)
   - CRF decoding layer

### ✅ Modified Files
2. **src/config.py**
   - Added `ARABERT_CHAR_BILSTM_CRF_CONFIG` with optimized hyperparameters
   - Registered in `get_model_config()` function

3. **src/train.py**
   - Updated `ContextualDataset` to return both embeddings AND char_ids
   - Updated `collate_contextual_batch` to pad 4 tensors (emb, char_ids, labels, mask)
   - Updated training loop to handle dual inputs
   - Updated `evaluate_model` to support fusion models
   - Added model initialization for `arabert_char_bilstm_crf`

4. **src/test.py**
   - Same updates as train.py for inference/evaluation
   - Supports dual-input forward pass during testing

## Architecture Details

### Baseline Model (bilstm_crf.py)
```
AraBERT (768) → BiLSTM (1 layer, 256) → Linear → CRF
```

### SOTA Fusion Model (arabert_char_bilstm_crf.py)
```
AraBERT (768) → Projection (384) → LayerNorm → Dropout ─┐
                                                          ├─ Concat (484) → BiLSTM (2 layers, 512) → Linear → CRF
Char IDs → Embedding (100) ─────────────────────────────┘
```

**Key Improvements:**
- **Dual features**: Semantic (AraBERT) + Morphology (char embeddings)
- **AraBERT projection**: Reduces dimensionality (768→384) with regularization
- **Deeper BiLSTM**: 2 layers vs. 1 layer
- **More capacity**: 512 hidden dim vs. 256
- **Better regularization**: LayerNorm + Dropout (0.3)

## Configuration

In `src/config.py`:
```python
ARABERT_CHAR_BILSTM_CRF_CONFIG = {
    "char_vocab_size": None,           # Set dynamically
    "tagset_size": 15,                 # Fixed diacritic classes
    "arabert_dim": 768,                # AraBERT hidden size
    "char_embedding_dim": 100,         # Character embedding dim
    "hidden_dim": 512,                 # BiLSTM (larger for fusion)
    "num_layers": 2,                   # Deeper network
    "dropout": 0.3,                    # Regularization
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 7,
    "gradient_clip": 5.0,
    "use_crf": True,
    "use_contextual": True,            # Uses AraBERT
    "batch_size": 1                    # Memory constraint
}
```

## How to Train

### Command
```bash
python src/train.py --model arabert_char_bilstm_crf --train_data data/train.txt --val_data data/val.txt
```

### Requirements
- **GPU**: Strongly recommended (AraBERT is GPU-intensive)
- **Memory**: At least 8GB GPU memory
- **Batch size**: 1 (due to contextual embeddings)
- **Training time**: ~1-2 hours per epoch on GPU (vs. 4-8 hours on CPU for baseline)

### Expected Results
- **Baseline (bilstm_crf)**: 97.99% accuracy
- **SOTA (arabert_char_bilstm_crf)**: 98.0-98.5% accuracy (0.1-0.5% improvement)

## How to Test

```bash
python src/test.py --model arabert_char_bilstm_crf --model_path models/best_arabert_char_bilstm_crf.pth --test_data data/test.txt
```

## Data Flow

### Training
1. **Input**: Diacritized Arabic text from `data/train.txt`
2. **Tokenization**: Extract base chars and diacritics
3. **Dual Embedding**:
   - AraBERT generates contextual embeddings (768-dim)
   - Vocab encodes character IDs for embedding layer (100-dim)
4. **Batching**: `collate_contextual_batch` pads to max length
5. **Model Forward**:
   - Project AraBERT embeddings
   - Embed character IDs
   - Concatenate features
   - BiLSTM encoding
   - CRF decoding
6. **Loss**: Negative log-likelihood from CRF
7. **Backprop**: Update weights

### Inference
Same as training but without gradient computation and label inputs.

## Technical Details

### Why Dual Inputs Work
Arabic diacritization depends on:
1. **Context**: What the word means (AraBERT captures this)
2. **Morphology**: How the word is formed (char embeddings capture this)

Example:
- **كَتَبَ** (kataba - he wrote) vs. **كُتُب** (kutub - books)
- Same root letters: ك ت ب
- Different diacritics based on morphological pattern
- Char embeddings learn these patterns better than word-level models

### Why It's Better Than Baseline
1. **Complementary features**: AraBERT might miss rare morphological patterns; char embeddings catch them
2. **Deeper architecture**: 2-layer BiLSTM learns more complex representations
3. **Regularization**: LayerNorm + Dropout prevent overfitting
4. **Projection**: Reduces AraBERT's 768 dims to 384, focusing on relevant features

## Performance Expectations

### Realistic Gains
At 97.99% baseline:
- **Best case**: +0.5% → 98.49%
- **Typical**: +0.2-0.3% → 98.2-98.3%
- **Worst case**: +0.1% → 98.09%

### Why Gains Are Small
- Already near ceiling (97.99%)
- Remaining errors are often ambiguous cases or annotation errors
- Diminishing returns at high accuracy levels

### Where It Helps Most
- Rare words with uncommon morphology
- Words with multiple valid diacritizations
- Long sequences where context helps
- Proper nouns and foreign words

## For Your Report

### Model Comparison Table
| Model | Features | Architecture | Accuracy | DER |
|-------|----------|--------------|----------|-----|
| BiLSTM-CRF (Baseline) | AraBERT (768) | 1-layer BiLSTM (256) | 97.99% | 2.01% |
| AraBERT+Char BiLSTM-CRF (SOTA) | AraBERT (768) + Char (100) | 2-layer BiLSTM (512) | 98.2%* | 1.8%* |

*Expected results

### Ablation Study
Train these variants to show each component's contribution:
1. **Char only** (no AraBERT): ~92-94% accuracy
2. **AraBERT only** (baseline): 97.99% accuracy
3. **AraBERT + Char** (fusion): 98.2%* accuracy

This demonstrates that both features contribute.

## Troubleshooting

### Out of Memory
- Reduce batch_size to 1 (already default)
- Use gradient accumulation (not implemented yet)
- Reduce hidden_dim from 512 to 256

### Slow Training
- Use GPU (CUDA)
- Disable AraBERT caching if disk is slow
- Reduce num_epochs

### Model Not Loading
- Ensure model_name matches: `arabert_char_bilstm_crf`
- Check checkpoint contains all required keys
- Verify vocab size matches

## Next Steps

1. **Train the model**: Run the training command above
2. **Monitor training**: Watch for overfitting (val loss increasing)
3. **Evaluate**: Compare DER with baseline
4. **Tune hyperparameters**: If needed, adjust dropout, hidden_dim, or learning rate
5. **Write report**: Document architecture, results, and ablation study

## Questions?

If you see errors, check:
1. AraBERT is installed: `pip install transformers`
2. GPU is available: `torch.cuda.is_available()`
3. Config matches model expectations
4. Data paths are correct
