# Arabic Diacritization System Architecture

## Overview

This document describes the complete architecture of our Arabic diacritization system, including data preprocessing, model architectures, training pipeline, and evaluation metrics.

## Table of Contents

1. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
2. [Model Architectures](#model-architectures)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Configuration Management](#configuration-management)
6. [AraBERT Alternative](#arabert-alternative)

## Data Preprocessing Pipeline

### 1. Raw Text Loading
- **Input**: Arabic text files with diacritics
- **Output**: Cleaned text lines
- **Process**:
  - Load training/validation/test files
  - Remove reference markers (e.g., `(20/8)`, `(1325)`)
  - Normalize Arabic punctuation
  - Filter non-Arabic characters

### 2. Tokenization (`src/preprocessing/tokenize.py`)
- **Function**: `tokenize_file()`
- **Input**: Cleaned Arabic text
- **Output**:
  - `X`: List of base character sequences (Arabic letters/digits)
  - `Y`: List of diacritic label sequences (strings)
  - `lines`: Original text lines
- **Process**:
  - Unicode normalization (NFKC → NFC)
  - Character-by-character processing
  - Diacritic canonicalization (Shadda first, then others)
  - Space skipping (implicit word boundaries)

### 3. Label Encoding (`src/preprocessing/encode_labels.py`)
- **Function**: `encode_corpus()`
- **Input**: Diacritic label strings
- **Output**: Integer label sequences
- **Mapping**: Uses `utils/diacritic2id.pickle`
  - 15 classes: 0-14 (empty string = no diacritic)
  - Examples: `''`→0, `'َ'`→1, `'ّ'`→2, etc.

### 4. Sequence Padding (`src/preprocessing/pad_sequences.py`)
- **Function**: `pad_sequences()`
- **Input**: Variable-length sequences
- **Output**: Fixed-length padded sequences
- **Parameters**:
  - `max_seq_length`: 256 (configurable)
  - Padding value: 0
  - Truncation: Keep leftmost characters

### 5. Vocabulary Building (`utils/vocab.py`)
- **Class**: `CharVocab`
- **Process**:
  - Build character-to-ID mapping from training data
  - Special tokens: `<PAD>`=0, `<UNK>`=1
  - Size: ~100-200 characters

## Model Architectures

### 1. BiLSTM-CRF Model (`src/models/bilstm_crf.py`)

#### Architecture Overview
```
Input Text → Character Embeddings → BiLSTM → CRF → Predictions
```

#### Components
- **Character Embeddings**:
  - With AraBERT: 768-dim contextual embeddings
  - Without AraBERT: 100-dim character embeddings
- **BiLSTM Encoder**:
  - Hidden dimension: 256 (128 per direction)
  - 1 layer, bidirectional
  - Dropout: 0.3
- **CRF Layer**:
  - 15 classes (diacritic types)
  - Viterbi decoding for inference
- **Loss**: Negative log-likelihood from CRF

#### Configuration
```python
BILSTM_CRF_CONFIG = {
    "vocab_size": None,  # Set dynamically
    "tagset_size": 15,
    "embedding_dim": 768,  # AraBERT
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.3,
    "use_crf": True,
    "use_contextual": True
}
```

### 2. Hierarchical BiLSTM Model (`src/models/hierarchical_bilstm.py`)

#### Architecture Overview
```
Input Text → Character Encoder → Word Feature Extraction → Combined Classifier → Predictions
                                      ↓
                            Convolution (3x1 kernel)
```

#### Components
- **Character-Level BiLSTM**:
  - Embedding dim: 128
  - Hidden dim: 256 (128 per direction)
  - 2 layers, bidirectional
  - Dropout: 0.3
- **Word-Level Feature Extraction**:
  - 1D Convolution (kernel_size=3, padding=1)
  - Input: Character features (256-dim)
  - Output: Word features (256-dim)
- **Combined Classifier**:
  - Input: 512-dim (char + word features)
  - Hidden: 512-dim with ReLU
  - Output: 15 classes (diacritics)
  - Dropout: 0.3
- **Loss**: Cross-entropy (no CRF)

#### Configuration
```python
HIERARCHICAL_BILSTM_CONFIG = {
    "char_vocab_size": None,
    "word_vocab_size": None,
    "char_embedding_dim": 128,
    "word_embedding_dim": 256,
    "char_hidden_dim": 256,
    "word_hidden_dim": 256,
    "char_num_layers": 2,
    "word_num_layers": 2,
    "classifier_hidden_dim": 512,
    "num_classes": 15,
    "dropout": 0.3
}
```

## Training Pipeline

### 1. Data Loading (`src/train.py`)
- **Function**: `load_data()`
- **Process**:
  - Load train/val files
  - Apply tokenization and encoding
  - Optional max_samples for debugging

### 2. Model Initialization
- **Function**: `get_model()`
- **Process**:
  - Load model config
  - Initialize appropriate model class
  - Move to device (CPU/GPU)

### 3. Training Loop
- **Components**:
  - Adam optimizer (lr=0.001, weight_decay=1e-5)
  - StepLR scheduler (step_size=10, gamma=0.5)
  - Gradient clipping (max_norm=5.0)
  - Early stopping (patience=7)
- **Batch Processing**:
  - Batch size: 32 (non-contextual), 1 (contextual)
  - Custom collate for contextual embeddings
- **Checkpointing**:
  - Save best model by validation DER
  - Complete checkpoint with optimizer/scheduler state

### 4. Validation
- **Function**: `evaluate_model()`
- **Metrics**:
  - Character-level accuracy (excluding spaces)
  - Diacritic Error Rate (DER)
- **Process**:
  - Model in eval mode
  - No gradient computation
  - CRF decoding for predictions

## Evaluation Metrics

### 1. Character Accuracy
- **Formula**: `correct_predictions / total_characters`
- **Exclusions**: Spaces and padding
- **Range**: 0.0 - 1.0

### 2. Diacritic Error Rate (DER)
- **Formula**: `errors / total_characters`
- **Definition**: Percentage of characters with wrong diacritics
- **Range**: 0.0 - 1.0 (lower is better)

### 3. Word Error Rate (WER) - Future
- **Formula**: `incorrect_words / total_words`
- **Process**: Compare full diacritized words

## Configuration Management

### File Structure
```
src/config.py
├── DATA_CONFIG          # Data loading settings
├── TRAINING_CONFIG      # Training hyperparameters
├── EVALUATION_CONFIG    # Evaluation settings
├── MODEL_CONFIGS        # Model-specific configs
│   ├── RNN_CONFIG
│   ├── LSTM_CONFIG
│   ├── CRF_CONFIG
│   ├── BILSTM_CRF_CONFIG
│   └── HIERARCHICAL_BILSTM_CONFIG
└── UTILITY_FUNCTIONS    # Config helpers
```

### Key Configurations

#### Data Configuration
```python
DATA_CONFIG = {
    "max_seq_length": 256,
    "batch_size": 32,
    "num_workers": 2
}
```

#### Training Configuration
```python
TRAINING_CONFIG = {
    "scheduler": "step",
    "step_size": 10,
    "gamma": 0.5
}
```

## AraBERT Alternative

### Architecture Overview
```
Input Text → AraBERT Encoder → BiLSTM → CRF → Predictions
```

### Components
- **AraBERT Encoder**:
  - Model: `aubmindlab/bert-base-arabertv02`
  - Output: 768-dim contextual embeddings
  - Frozen weights (feature extraction only)
- **BiLSTM Layer**:
  - Input: AraBERT embeddings (768-dim)
  - Hidden: 256-dim (128 per direction)
  - 1 layer, bidirectional
- **CRF Layer**:
  - 15 diacritic classes
  - Transition scores learned
- **Training**:
  - Only BiLSTM + CRF parameters trained
  - AraBERT weights frozen

### Advantages
- **Contextual Understanding**: AraBERT captures Arabic morphology and context
- **Transfer Learning**: Pre-trained on large Arabic corpus
- **Better Representations**: 768-dim vs 100-dim character embeddings

### Configuration
```python
ARABERT_BILSTM_CRF_CONFIG = {
    "vocab_size": None,
    "tagset_size": 15,
    "embedding_dim": 768,  # AraBERT hidden size
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "use_crf": True,
    "use_contextual": True,
    "freeze_bert": True  # Freeze AraBERT weights
}
```

### Implementation Notes
- **Memory Usage**: AraBERT requires significant GPU memory
- **Batch Size**: Limited to 1-2 for long sequences
- **Training Time**: Longer due to embedding computation
- **Performance**: Expected 90-95% accuracy vs 85-90% for character embeddings

## File Structure Summary

```
src/
├── config.py              # Configuration management
├── train.py               # Main training script
├── test.py                # Testing and evaluation
├── preprocessing/
│   ├── tokenize.py        # Text tokenization
│   ├── encode_labels.py   # Label encoding
│   └── pad_sequences.py   # Sequence padding
├── models/
│   ├── bilstm_crf.py      # BiLSTM-CRF model
│   └── hierarchical_bilstm.py  # Hierarchical model
├── features/
│   └── contextual_embeddings.py  # AraBERT integration
└── utils/
    └── vocab.py           # Vocabulary management

utils/
├── diacritic2id.pickle    # Diacritic mapping
└── vocab.py              # Character vocabulary

models/                    # Saved model checkpoints
├── best_bilstm_crf.pth
└── best_hierarchical_bilstm.pth

data/                     # Training data
├── train.txt
├── val.txt
└── processed/           # Preprocessed data cache
```

## Performance Expectations

### BiLSTM-CRF with AraBERT
- **Accuracy**: 90-95%
- **DER**: 5-10%
- **Training Time**: 2-4 hours
- **GPU Memory**: 4-8GB

### Hierarchical BiLSTM
- **Accuracy**: 85-90%
- **DER**: 10-15%
- **Training Time**: 1-2 hours
- **GPU Memory**: 2-4GB

### Character-Level BiLSTM-CRF
- **Accuracy**: 85-90%
- **DER**: 10-15%
- **Training Time**: 30-60 minutes
- **GPU Memory**: 1-2GB