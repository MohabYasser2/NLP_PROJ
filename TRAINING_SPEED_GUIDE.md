# Training Speed Optimization Guide

## Speed Optimizations Implemented

### 1. **Validation Frequency Control**
- **Setting**: `EVALUATION_CONFIG['validation_frequency']` in `src/config.py`
- **Default**: `1` (every epoch)
- **Fast Training**: Set to `2-5` (evaluate every 2-5 epochs)
- **Impact**: 2-5x faster training by skipping expensive validation

### 2. **Validation Subset**
- **Setting**: `EVALUATION_CONFIG['validation_sample_size']` in `src/config.py`
- **Default**: `None` (use all validation data)
- **Fast Training**: Set to `500-1000` (use subset of validation)
- **Impact**: 2-5x faster validation (proportional to subset size)

### 3. **Skip Early Validation**
- **Setting**: `EVALUATION_CONFIG['eval_start_epoch']` in `src/config.py`
- **Default**: `1` (start immediately)
- **Fast Training**: Set to `5-10` (skip first epochs)
- **Impact**: Saves time in early training when accuracy is poor anyway

### 4. **Cached Validation Embeddings**
- **Automatic**: When using AraBERT (`use_contextual=True`)
- **Impact**: 10-20x faster validation (embeddings computed once, not every epoch)
- **Memory**: Uses ~2GB RAM for 2500 validation samples

### 5. **Checkpoint Saving**
- **Every Epoch**: Saves `models/epoch_N_modelname.pth`
- **Best Model**: Updates `models/best_modelname.pth` when DER improves
- **Benefit**: Can resume from any epoch, not just best

## Recommended Settings

### For Maximum Speed (Development/Testing)
```python
# In src/config.py - EVALUATION_CONFIG
"validation_frequency": 5,          # Validate every 5 epochs
"validation_sample_size": 500,      # Use only 500 samples
"eval_start_epoch": 5               # Skip first 5 epochs
```
**Expected speedup**: 10-15x faster training

### For Balanced (Training on Kaggle)
```python
# In src/config.py - EVALUATION_CONFIG
"validation_frequency": 2,          # Validate every 2 epochs
"validation_sample_size": 1000,     # Use 1000 samples
"eval_start_epoch": 3               # Skip first 3 epochs
```
**Expected speedup**: 4-6x faster training

### For Full Accuracy (Final Model)
```python
# In src/config.py - EVALUATION_CONFIG
"validation_frequency": 1,          # Validate every epoch
"validation_sample_size": None,     # Use all validation data
"eval_start_epoch": 1               # Validate from start
```
**Expected speedup**: 2x faster (from cached embeddings only)

## Training Time Estimates

### Original (No Optimizations)
- **AraBERT model**: ~2-3 hours for 50 epochs on RTX 2070
- **Validation**: ~8-10 minutes per epoch (2500 samples)

### With All Optimizations (Maximum Speed)
- **AraBERT model**: ~15-20 minutes for 50 epochs
- **Validation**: ~30 seconds per epoch (500 samples, every 5 epochs)

## How to Use

1. **Edit** `src/config.py` - modify `EVALUATION_CONFIG` settings
2. **Run training** as usual:
   ```bash
   python -m src.train --model arabert_char_bilstm_crf --train_data data/train.txt --val_data data/val.txt
   ```
3. **Check speed summary** at training start - shows all active optimizations
4. **Monitor checkpoints** in `models/` folder - saved every epoch

## Tips

- **During development**: Use maximum speed settings to iterate quickly
- **Before final training**: Switch to full accuracy settings
- **On Kaggle**: Use balanced settings to fit within time limits
- **Memory issues?**: Reduce `validation_sample_size` to 250-500

## Technical Details

### Validation Embedding Cache
- Computed once at training start (before epoch 1)
- Stored in memory (`val_embeddings_cache`, `val_char_ids_cache`)
- Automatically used during all validation runs
- Falls back to on-the-fly computation if cache unavailable

### Checkpoint Format
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_der': best_der_so_far,
    'val_der': current_epoch_der,
    'train_loss': current_epoch_loss,
    'config': model_config,
    'vocab': vocabulary_mapping
}
```

### Memory Usage
- **Training embeddings**: Pre-computed in `ContextualDataset.__init__()` (~10GB for 43k samples)
- **Validation cache**: Computed once (~2GB for 2.5k samples)
- **Total**: ~12-15GB GPU memory for full training

## Troubleshooting

**Q: Validation accuracy seems wrong?**
- Ensure `validation_sample_size` is representative (>500 samples)
- Use `None` for final evaluation to get true accuracy

**Q: Out of memory during training?**
- Validation cache is optional - will fallback to on-the-fly
- Reduce batch size in model config
- Reduce validation subset size

**Q: Want to resume from specific epoch?**
- Load checkpoint: `models/epoch_N_modelname.pth`
- Contains full training state for resumption
