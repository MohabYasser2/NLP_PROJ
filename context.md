# Project Context

This file serves as a persistent context for the Arabic Diacritization project session. It will be updated throughout the development process to track key information, decisions, and current state.

## Project Overview
Arabic diacritization project for NLP course. The goal is to restore missing diacritics in Arabic text using machine learning approaches.

## Current File Structure
```
.
├── context.md
├── data/
│   ├── processed/
│   ├── raw/
│   ├── train.txt
│   └── val.txt
├── models/
├── notebooks/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── features/
│   │   ├── bag_of_words.py
│   │   ├── char_features.py
│   │   ├── contextual_embeddings.py
│   │   ├── extract_features.py
│   │   ├── prefix_suffix.py
│   │   ├── tfidf.py
│   │   ├── word_embeddings.py
│   │   └── word_features.py
│   ├── models/
│   │   ├── bilstm_crf.py
│   │   ├── crf.py
│   │   ├── lstm.py
│   │   └── rnn.py
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── encode_labels.py
│   │   ├── pad_sequences.py
│   │   └── tokenize.py
│   ├── test.py
│   ├── train.py ✅ (Complete training script with progress bars and metrics)
│   └── train_bilstm_crf.py
├── test_input.txt
├── test_output.txt
└── utils/
    ├── arabic_letters.pickle
    ├── diacritic2id.pickle
    ├── diacritics.pickle
    └── vocab.py
```

## Utils Files Contents

### helpers.py
```python
# Placeholder for utility functions
```

### arabic_letters.pickle
A set containing Arabic letters:
```
{'أ', 'خ', 'ق', 'إ', 'ز', 'ك', 'د', 'ط', 'م', 'ر', 'ح', 'ن', 'ئ', 'ي', 'ا', 'آ', 'ء', 'ش', 'ض', 'ذ', 'ب', 'ؤ', 'ة', 'ت', 'ظ', 'غ', 'ص', 'س', 'و', 'ج', 'ع', 'ف', 'ث', 'ه', 'ى', 'ل'}
```

### diacritic2id.pickle
A dictionary mapping diacritics to IDs:
```
{'َ': 0, 'ً': 1, 'ُ': 2, 'ٌ': 3, 'ِ': 4, 'ٍ': 5, 'ْ': 6, 'ّ': 7, 'َّ': 8, 'ًّ': 9, 'ُّ': 10, 'ٌّ': 11, 'ِّ': 12, 'ٍّ': 13, '': 14}
```

### diacritics.pickle
A set containing diacritics:
```
{'ّ', 'ُ', 'ٌ', 'َ', 'ِ', 'ْ', 'ٍ', 'ً'}
```

## Current Status
- File structure created with placeholders
- Utils files contain pre-existing data for Arabic letters and diacritics
- clean_data.py implemented and tested successfully
- Ready to implement preprocessing, features, models, training, and testing

## Updates
- [Date/Time]: Initial context established
- [December 14, 2025]: Tested clean_data.py - works correctly, preserves diacritics while removing noise
- [December 14, 2025]: Tested tokenize.py - successfully tokenizes Arabic text into base characters and diacritic labels
- [December 14, 2025]: Tested encode_labels.py - successfully encodes diacritic labels into integer IDs using the mapping
- [December 14, 2025]: Implemented and tested feature extraction modules (char_features, word_features, prefix_suffix, extract_features)
- [December 14, 2025]: Implemented BiLSTM-CRF model components (vocab.py, bilstm_crf.py, train_bilstm_crf.py) and tested core utilities
- [December 14, 2025]: Installed PyTorch dependencies (torch, TorchCRF) - model ready for training- [December 14, 2025]: Successfully tested BiLSTM-CRF model instantiation and inference
- [December 14, 2025]: Created comprehensive config.py with hyperparameters for all models
- [December 14, 2025]: Implemented complete training script (src/train.py) with model selection, progress bars, accuracy/DER metrics, and early stopping - metrics display fixed and tested successfully
- [December 14, 2025]: Fixed all 8 critical issues: data leakage prevention, diacritic mapping consistency, CRF batch handling, loss computation, accuracy calculation (excluding spaces), complete checkpoint saving, reproducibility controls, and proper imports