"""
Configuration file for Arabic Diacritization Models
Contains hyperparameters and settings for all implemented models
"""

# ======================================================
# General Settings
# ======================================================

DATA_CONFIG = {
    "max_seq_length": 256,  # Maximum sequence length for batching
    "batch_size": 32,  # ✅ Updated from 1 for efficiency
    "num_workers": 2,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}

# Diacritic classes (from diacritic2id.pickle)
NUM_DIACRITIC_CLASSES = 15

# ======================================================
# RNN Model Configuration
# ======================================================

RNN_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "embedding_dim": 100,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 5,  # Early stopping patience
    "gradient_clip": 5.0
}

# ======================================================
# LSTM Model Configuration
# ======================================================

LSTM_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "embedding_dim": 100,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 5,
    "gradient_clip": 5.0
}

# ======================================================
# CRF Model Configuration
# ======================================================

CRF_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "embedding_dim": 100,
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "num_epochs": 30,
    "patience": 5,
    "gradient_clip": 1.0,
    "use_crf": True
}

# ======================================================
# BiLSTM-CRF Model Configuration
# ======================================================

BILSTM_CRF_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "tagset_size": NUM_DIACRITIC_CLASSES,
    "embedding_dim": 768,  # AraBERT hidden size when using contextual
    "hidden_dim": 256,  # BiLSTM so effective hidden is 128 per direction
    "num_layers": 1,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 7,
    "gradient_clip": 5.0,
    "use_crf": True,
    "use_contextual": True  # Use AraBERT embeddings
}

# ======================================================
# BiLSTM Classifier Configuration (Dual-Pathway)
# ======================================================

BILSTM_CLASSIFIER_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "tagset_size": NUM_DIACRITIC_CLASSES,
    "embedding_dim": 768,  # AraBERT hidden size when using contextual
    "hidden_dim": 256,  # Not directly used, for compatibility
    "char_emb_dim": 128,  # Character embedding dimension (if not contextual)
    "char_hidden_dim": 128,  # Character BiLSTM hidden dimension
    "word_hidden_dim": 128,  # Word BiLSTM hidden dimension
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 10,
    "patience": 7,
    "gradient_clip": 5.0,
    "use_crf": False,  # No CRF in this variant
    "use_contextual": True  # Use AraBERT embeddings
}

# ======================================================
# BiLSTM Classifier + CRF Configuration (Dual-Pathway with CRF)
# ======================================================

BILSTM_CLASSIFIER_CRF_CONFIG = {
    "vocab_size": None,  # Will be set dynamically based on data
    "tagset_size": NUM_DIACRITIC_CLASSES,
    "embedding_dim": 768,  # AraBERT hidden size when using contextual
    "hidden_dim": 256,  # Not directly used, for compatibility
    "char_emb_dim": 128,  # Character embedding dimension (if not contextual)
    "char_hidden_dim": 128,  # Character BiLSTM hidden dimension
    "word_hidden_dim": 128,  # Word BiLSTM hidden dimension
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 7,
    "gradient_clip": 5.0,
    "use_crf": True,  # WITH CRF in this variant
    "use_contextual": True  # Use AraBERT embeddings
}

# ======================================================
# CONFIGURATION NOTES FOR HIGH ACCURACY
# ======================================================
# 
# For CPU Training (Local Machine):
#   - embedding_dim: 100 (character embeddings)
#   - use_contextual: False
#   - Expected accuracy: 85-90%
#   - Training time: 4-8 hours for 100 epochs
#
# For GPU Training (Kaggle):
#   - embedding_dim: 768 (AraBERT embeddings)
#   - use_contextual: True
#   - Expected accuracy: 90-95%
#   - Training time: 12-20 hours for 100 epochs
#
# To switch: Change embedding_dim and use_contextual together

# ======================================================
# Feature-Based Models (if using sklearn-style models)
# ======================================================

FEATURE_MODEL_CONFIG = {
    "crf": {
        "algorithm": "lbfgs",
        "c1": 0.1,
        "c2": 0.1,
        "max_iterations": 100,
        "all_possible_transitions": True
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
}

# ======================================================
# Training Configuration
# ======================================================

TRAINING_CONFIG = {
    "optimizer": "adam",
    "scheduler": "step",  # "step", "cosine", "plateau"
    "step_size": 10,
    "gamma": 0.5,
    "warmup_steps": 100,
    "save_best_only": True,
    "save_path": "models/",
    "log_interval": 10,
    "eval_interval": 1
}

# ======================================================
# Evaluation Configuration
# ======================================================

EVALUATION_CONFIG = {
    "metrics": ["der", "wer", "accuracy"],  # Diacritic Error Rate, Word Error Rate, Accuracy
    "der_threshold": 0.1,  # Consider prediction correct if DER < threshold
    "save_predictions": True,
    "prediction_path": "predictions/"
}

# ======================================================
# Utility Functions
# ======================================================

def get_model_config(model_name: str):
    """Get configuration for a specific model"""
    configs = {
        "rnn": RNN_CONFIG,
        "lstm": LSTM_CONFIG,
        "crf": CRF_CONFIG,
        "bilstm_crf": BILSTM_CRF_CONFIG,
        "bilstm_classifier": BILSTM_CLASSIFIER_CONFIG,
        "bilstm_classifier_crf": BILSTM_CLASSIFIER_CRF_CONFIG
    }
    return configs.get(model_name.lower())

def update_vocab_size(config: dict, vocab_size: int):
    """Update vocab_size in model config"""
    config["vocab_size"] = vocab_size
    return config

def get_feature_model_config(model_name: str):
    """Get configuration for feature-based models"""
    return FEATURE_MODEL_CONFIG.get(model_name.lower())