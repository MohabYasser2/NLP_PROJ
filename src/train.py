#!/usr/bin/env python3
"""
Arabic Diacritization Training Script

Supports multiple models: RNN, LSTM, CRF, BiLSTM-CRF
Uses configuration from config.py
Shows progress bar and evaluation metrics
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContextualDataset(Dataset):
    """
    On-the-fly embedding dataset.
    Embeddings are computed dynamically in __getitem__ to save memory.
    """
    def __init__(self, X, Y, lines, vocab, config, diacritic2id, embedder):
        self.lines = lines
        self.vocab = vocab
        self.config = config
        self.diacritic2id = diacritic2id
        self.embedder = embedder
        
        # Pre-encode labels for efficiency
        self.Y_encoded = encode_corpus(Y, diacritic2id)
        
        print(f"✓ Dataset initialized with {len(lines)} samples (on-the-fly embedding)")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        
        # Extract base characters (tokenize_line handles diacritics correctly)
        base_chars, _ = tokenize_line(line)
        
        # Skip empty sequences
        if not base_chars:
            # Return minimal valid item
            return {
                'embedding': torch.zeros((1, 768), dtype=torch.float32),
                'char_ids': torch.tensor([0], dtype=torch.long),
                'label': torch.tensor([0], dtype=torch.long),
                'mask': torch.tensor([False], dtype=torch.bool)
            }
        
        # Compute embedding on-the-fly
        emb = self.embedder.embed_line_chars(line)
        char_ids = self.vocab.encode(base_chars)
        y_seq = self.Y_encoded[idx]
        
        # Align lengths
        T = min(len(emb), len(char_ids), len(y_seq))
        
        return {
            'embedding': torch.tensor(emb[:T], dtype=torch.float32),      # (T, 768)
            'char_ids': torch.tensor(char_ids[:T], dtype=torch.long),     # (T,)
            'label': torch.tensor(y_seq[:T], dtype=torch.long),           # (T,)
            'mask': torch.tensor([True] * T, dtype=torch.bool)            # (T,)
        }


def collate_contextual_batch(batch):
    """
    Collate function to pad embeddings, char_ids, labels, and masks in a batch.
    Supports dual-input models (AraBERT + char morphology).
    """
    embeddings = [item['embedding'] for item in batch]
    char_ids = [item['char_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    # Find max length in this batch
    max_len = max(len(emb) for emb in embeddings)
    
    # Pad all to max_len
    padded_embeddings = []
    padded_char_ids = []
    padded_labels = []
    padded_masks = []
    
    for emb, ch_id, label, mask in zip(embeddings, char_ids, labels, masks):
        pad_len = max_len - len(emb)
        if pad_len > 0:
            # Pad embeddings (T, 768) -> (max_len, 768)
            padded_emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len), value=0.0)
            # Pad char_ids (T,) -> (max_len,)
            padded_ch = torch.nn.functional.pad(ch_id, (0, pad_len), value=0)
            # Pad labels (T,) -> (max_len,)
            padded_label = torch.nn.functional.pad(label, (0, pad_len), value=0)
            # Pad masks (T,) -> (max_len,)
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_emb = emb
            padded_ch = ch_id
            padded_label = label
            padded_mask = mask
        
        padded_embeddings.append(padded_emb)
        padded_char_ids.append(padded_ch)
        padded_labels.append(padded_label)
        padded_masks.append(padded_mask)
    
    return (
        torch.stack(padded_embeddings),  # (batch, max_len, 768)
        torch.stack(padded_char_ids),    # (batch, max_len)
        torch.stack(padded_labels),      # (batch, max_len)
        torch.stack(padded_masks)        # (batch, max_len)
    )

# Import our modules
from src.preprocessing.tokenize import tokenize_file, tokenize_line
from src.preprocessing.encode_labels import encode_corpus
from utils.vocab import CharVocab
from src.config import (
    get_model_config, update_vocab_size,
    DATA_CONFIG, TRAINING_CONFIG, EVALUATION_CONFIG
)
from src.preprocessing.pad_sequences import pad_sequences
from src.features.contextual_embeddings import ContextualEmbedder
from src.features.ngram_features import NgramExtractor

# Import models
from src.models.bilstm_crf import BiLSTMCRF
# from src.models.arabert_bilstm_crf import AraBERTBiLSTMCRF
from src.models.arabert_char_bilstm_crf import AraBERTCharBiLSTMCRF
from src.models.char_bilstm_classifier import CharBiLSTMClassifier
from src.models.charngram_bilstm_classifier import CharNgramBiLSTMClassifier
# TODO: Import other models when implemented
# from src.models.rnn import RNNModel
# from src.models.lstm import LSTMModel
# from src.models.crf import CRFModel


def load_data(train_path, val_path, max_samples=None):
    """Load and preprocess training/validation data"""
    print("Loading training data...")
    X_train, Y_train, lines_train = tokenize_file(train_path)
    if max_samples:
        X_train = X_train[:max_samples]
        Y_train = Y_train[:max_samples]
        lines_train = lines_train[:max_samples]

    print("Loading validation data...")
    X_val, Y_val, lines_val = tokenize_file(val_path)
    if max_samples:
        X_val = X_val[:max_samples]
        Y_val = Y_val[:max_samples]
        lines_val = lines_val[:max_samples]

    return X_train, Y_train, lines_train, X_val, Y_val, lines_val


def prepare_data(X, Y, lines, vocab, config, diacritic2id, embedder=None, ngram_extractor=None):
    """Prepare data for training"""
    if config.get("use_contextual", False):
        # Use custom dataset that computes embeddings on-the-fly
        print("Preparing dataset (embeddings computed on-the-fly)...")
        dataset = ContextualDataset(X, Y, lines, vocab, config, diacritic2id, embedder)
    else:
        # Encode sequences
        X_encoded = [vocab.encode(seq) for seq in X]

        # Pad sequences
        X_padded, mask = pad_sequences(X_encoded, pad_value=vocab.char2id["<PAD>"])

        # Encode labels using the loaded diacritic mapping
        Y_encoded = encode_corpus(Y, diacritic2id)

        Y_padded, _ = pad_sequences(Y_encoded, pad_value=0)  # Use 0 for padding (valid tag index)

        # Extract n-gram features if needed
        if ngram_extractor is not None:
            # Extract and encode n-grams for each sequence
            ngram_encoded = [ngram_extractor.encode(seq) for seq in X]
            ngram_padded, _ = pad_sequences(ngram_encoded, pad_value=0)  # Pad with 0
            # Create dataset with n-grams
            dataset = TensorDataset(X_padded, ngram_padded, Y_padded, mask)
        else:
            # Create dataset without n-grams
            dataset = TensorDataset(X_padded, Y_padded, mask)

    return dataset


def get_model(model_name, config):
    """Initialize the specified model"""
    if model_name.lower() == "bilstm_crf":
        # Filter config to only include parameters that BiLSTMCRF accepts
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "use_contextual": config.get("use_contextual", False)
        }
        model = BiLSTMCRF(**model_config)
    elif model_name.lower() == "hierarchical_bilstm":
        # For hierarchical model, we need char and word vocab sizes
        model_config = config.copy()
        model_config["char_vocab_size"] = config["char_vocab_size"]
        model_config["word_vocab_size"] = config["word_vocab_size"]
        # model = HierarchicalBiLSTM(model_config)
    elif model_name.lower() == "arabert_bilstm_crf":
        # AraBERT model
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "freeze_arabert": config.get("freeze_arabert", True)
        }
        # model = AraBERTBiLSTMCRF(**model_config)
    elif model_name.lower() == "arabert_char_bilstm_crf":
        # AraBERT + Character Fusion model (SOTA)
        model_config = {
            "char_vocab_size": config["char_vocab_size"],
            "tagset_size": config["tagset_size"],
            "arabert_dim": config["arabert_dim"],
            "char_embedding_dim": config["char_embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"]
        }
        model = AraBERTCharBiLSTMCRF(**model_config)
    elif model_name.lower() == "char_bilstm_classifier":
        # Character-only BiLSTM Classifier (Simple)
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"]
        }
        model = CharBiLSTMClassifier(**model_config)
    elif model_name.lower() == "charngram_bilstm_classifier":
        # Character + N-gram BiLSTM Classifier (Improved)
        model_config = {
            "char_vocab_size": config["char_vocab_size"],
            "ngram_vocab_size": config["ngram_vocab_size"],
            "tagset_size": config["tagset_size"],
            "char_embedding_dim": config["char_embedding_dim"],
            "ngram_embedding_dim": config["ngram_embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"]
        }
        model = CharNgramBiLSTMClassifier(**model_config)
    else:
        raise ValueError(f"Model {model_name} not implemented yet")

    return model


def calculate_der(predictions, targets, mask):
    """Calculate Diacritic Error Rate"""
    total_chars = 0
    errors = 0

    for pred_seq, target_seq, mask_seq in zip(predictions, targets, mask):
        for p, t, m in zip(pred_seq, target_seq, mask_seq):
            if m:  # Only count non-padded positions
                total_chars += 1
                if p != t:
                    errors += 1

    return errors / total_chars if total_chars > 0 else 0


def evaluate_model(model, lines, Y, vocab, embedder, device, diacritic2id, model_name="bilstm_crf", config=None):
    """
    Evaluate model on validation set using COMPETITION MODE CSV comparison.
    Generates CSV predictions and compares against golden CSV for exact competition accuracy.
    
    Embeddings are computed on-the-fly to save memory.
    """
    model.eval()
    
    # Check model type
    is_fusion_model = model_name.lower() == "arabert_char_bilstm_crf"
    is_ngram_classifier = model_name.lower() == "charngram_bilstm_classifier"
    is_simple_classifier = model_name.lower() == "char_bilstm_classifier"
    
    # Pre-encode all golden labels to create golden CSV
    Y_encoded = encode_corpus(Y, diacritic2id, skip_unknown=True)
    
    golden_csv = []  # List of (char_id, label) tuples
    prediction_csv = []  # List of (char_id, label) tuples
    char_id = 0

    with torch.no_grad():
        for line_idx, line in enumerate(tqdm(lines, desc="Evaluating")):
            # Get base characters and golden diacritics using tokenize_line
            base_chars, golden_diacritics = tokenize_line(line)
            
            if not base_chars:
                continue
            
            # Create golden CSV entries
            gold_labels = Y_encoded[line_idx][:len(base_chars)]
            for label in gold_labels:
                golden_csv.append((char_id, int(label)))
                char_id += 1
            
            # Reset char_id for predictions (will match golden)
            pred_start_id = char_id - len(gold_labels)
            
            # Prepare input based on model type
            if config.get("use_contextual", False):
                # Compute embeddings on-the-fly
                emb = embedder.embed_line_chars(line)
                char_ids = vocab.encode(base_chars)
                min_len = min(len(emb), len(char_ids))
                emb = emb[:min_len]
                char_ids = char_ids[:min_len]
                
                # For fusion models, we need matching char_ids
                if is_fusion_model:
                    X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    char_ids_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
                    mask = torch.tensor([True] * len(emb), dtype=torch.bool).unsqueeze(0).to(device)
                else:
                    X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    mask = torch.tensor([True] * len(emb), dtype=torch.bool).unsqueeze(0).to(device)
            else:
                # Encode characters
                X_encoded = vocab.encode(base_chars)
                X_padded, mask_tensor = pad_sequences([X_encoded], pad_value=vocab.char2id["<PAD>"])
                X_tensor = X_padded.to(device)
                mask = mask_tensor.to(device)
            
            # Predict
            if config.get("use_contextual", False) and is_fusion_model:
                pred = model(X_tensor, char_ids_tensor, mask=mask)
            elif is_ngram_classifier:
                char_ids = vocab.encode(base_chars)
                char_ids_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
                ngram_ids_tensor = char_ids_tensor.clone()
                _, pred = model(char_ids_tensor, ngram_ids_tensor, mask=mask)
            elif is_simple_classifier:
                _, pred = model(X_tensor, mask=mask)
            else:
                pred = model(X_tensor, mask=mask)
            
            # Extract predictions
            if isinstance(pred, list):
                # Handle different CRF return formats
                if len(pred) > 0 and isinstance(pred[0], list):
                    if len(pred) == 1:
                        pred_labels = pred[0][:len(base_chars)]
                    else:
                        pred_labels = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in pred]
                        pred_labels = pred_labels[:len(base_chars)]
                else:
                    pred_labels = pred[:len(base_chars)]
            else:
                pred_labels = pred[0][:len(base_chars)].cpu().tolist()
            
            # Convert tensor predictions to integers and create prediction CSV entries
            for idx, p in enumerate(pred_labels):
                label = int(p.item()) if isinstance(p, torch.Tensor) else int(p)
                prediction_csv.append((pred_start_id + idx, label))

    # Calculate accuracy by comparing CSV entries (exactly like competition)
    if len(golden_csv) != len(prediction_csv):
        print(f"WARNING: Length mismatch! Golden: {len(golden_csv)}, Predictions: {len(prediction_csv)}")
        min_len = min(len(golden_csv), len(prediction_csv))
    else:
        min_len = len(golden_csv)
    
    correct = 0
    total = 0
    
    for i in range(min_len):
        golden_id, golden_label = golden_csv[i]
        pred_id, pred_label = prediction_csv[i]
        
        if golden_id != pred_id:
            print(f"WARNING: ID mismatch at position {i}! Golden: {golden_id}, Pred: {pred_id}")
        
        total += 1
        if golden_label == pred_label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    der = 1.0 - accuracy

    return accuracy, der


def train_model(model_name, train_path, val_path, max_samples=None, seed=42):
    """Main training function"""
    # Set seeds for reproducibility
    set_seed(seed)

    # Get configuration
    config = get_model_config(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Training {model_name.upper()} model with seed {seed}")
    print(f"Embedding: {'AraBERT (768-dim)' if config.get('use_contextual') else 'Character (100-dim)'}")
    print(f"CRF: {'Enabled' if config.get('use_crf') else 'Disabled'}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    print("\nLoading data...")
    X_train, Y_train, lines_train, X_val, Y_val, lines_val = load_data(train_path, val_path, max_samples)
    print(f"✓ Loaded {len(X_train)} training samples")
    print(f"✓ Loaded {len(X_val)} validation samples")

    # Load diacritic mapping from pickle file (single source of truth)
    with open("utils/diacritic2id.pickle", "rb") as f:
        diacritic2id = pickle.load(f)
    print(f"✓ Loaded {len(diacritic2id)} diacritic classes")

    # Initialize embedder if using contextual embeddings
    embedder = None
    if config.get("use_contextual", False):
        print("\nInitializing AraBERT embedder...")
        embedder = ContextualEmbedder(
            model_name="aubmindlab/bert-base-arabertv02",
            device=device.type,
            cache_dir=None  # Disable disk caching (use in-memory only) to avoid Kaggle disk space issues
        )
        # Update embedding_dim for contextual
        config["embedding_dim"] = embedder.hidden_size
        print(f"✓ AraBERT loaded (hidden_size={embedder.hidden_size})")

    # Build vocabulary from training data ONLY (no data leakage)
    print("\nBuilding vocabulary...")
    vocab = CharVocab()
    vocab.build(X_train)
    print(f"✓ Vocabulary size: {len(vocab.char2id)}")

    # Update vocab size in config
    config = update_vocab_size(config.copy(), len(vocab.char2id))

    print(f"Updated config with vocab_size: {config['vocab_size']}")

    # Build n-gram vocabulary if needed
    ngram_extractor = None
    if model_name.lower() == "charngram_bilstm_classifier":
        print("\nBuilding n-gram vocabulary...")
        ngram_extractor = NgramExtractor(n=2)  # Use bigrams
        ngram_extractor.build_vocab(X_train)
        print(f"✓ N-gram vocabulary size: {len(ngram_extractor.ngram2id)}")
        
        # Update config with n-gram vocab size
        from src.config import update_ngram_vocab_size
        config = update_ngram_vocab_size(config, len(ngram_extractor.ngram2id))
        print(f"Updated config with ngram_vocab_size: {config['ngram_vocab_size']}")

    # Prepare datasets
    train_dataset = prepare_data(X_train, Y_train, lines_train, vocab, config, diacritic2id, embedder, ngram_extractor)
    val_dataset = prepare_data(X_val, Y_val, lines_val, vocab, config, diacritic2id, embedder, ngram_extractor)

    # Create dataloaders
    # Note: batch_size should be reasonable when using contextual embeddings
    batch_size = DATA_CONFIG['batch_size'] if not config.get("use_contextual", False) else 1
    
    # Use custom collate function for contextual embeddings
    collate_fn = collate_contextual_batch if config.get("use_contextual", False) else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 for Kaggle/notebooks (avoids multiprocessing issues)
        pin_memory=True  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    model = get_model(model_name, config)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0)
    )

    if TRAINING_CONFIG["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=TRAINING_CONFIG["step_size"],
            gamma=TRAINING_CONFIG["gamma"]
        )
    else:
        scheduler = None

    # Training loop
    best_der = float('inf')
    patience = config["patience"]
    patience_counter = 0
    
    # Print speed optimization summary
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"✓ Batch size: {config.get('batch_size', 'default')}")
    if config.get("use_contextual", False):
        print(f"✓ On-the-fly embeddings: {len(train_dataset)} training samples")
    validation_frequency = EVALUATION_CONFIG.get('validation_frequency', 1)
    validation_sample_size = EVALUATION_CONFIG.get('validation_sample_size', None)
    eval_start_epoch = EVALUATION_CONFIG.get('eval_start_epoch', 1)
    print(f"✓ Validation frequency: Every {validation_frequency} epoch(s)")
    if validation_sample_size:
        print(f"✓ Validation subset: {validation_sample_size}/{len(lines_val)} samples ({validation_sample_size/len(lines_val)*100:.1f}%)")
    print(f"✓ Evaluation starts at epoch: {eval_start_epoch}")
    print(f"✓ Checkpoint saving: Every epoch + best model")
    print("="*70 + "\n")
    
    # Check model type for appropriate handling
    is_fusion_model = model_name.lower() == "arabert_char_bilstm_crf"
    is_ngram_classifier = model_name.lower() == "charngram_bilstm_classifier"
    is_simple_classifier = model_name.lower() == "char_bilstm_classifier"

    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in progress_bar:
            if is_fusion_model:
                # Fusion model: unpack 4 tensors (embedding, char_ids, labels, mask)
                X_batch, char_ids_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                char_ids_batch = char_ids_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                optimizer.zero_grad()

                # Forward pass with dual inputs
                loss = model(X_batch, char_ids_batch, tags=y_batch, mask=mask_batch)
            elif is_ngram_classifier:
                # N-gram classifier: unpack 4 tensors (char_ids, ngram_ids, labels, mask)
                char_ids_batch, ngram_ids_batch, y_batch, mask_batch = batch
                char_ids_batch = char_ids_batch.to(device)
                ngram_ids_batch = ngram_ids_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                optimizer.zero_grad()

                # Forward pass with char and ngram inputs
                _, loss = model(char_ids_batch, ngram_ids_batch, tags=y_batch, mask=mask_batch)
            elif is_simple_classifier:
                # Simple classifier: unpack 3 tensors (char_ids, labels, mask)
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                optimizer.zero_grad()

                # Forward pass returns (logits, loss)
                _, loss = model(X_batch, tags=y_batch, mask=mask_batch)
            else:
                # Standard CRF model: unpack 3 tensors (X, y, mask)
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                optimizer.zero_grad()

                # Forward pass with single input
                loss = model(X_batch, tags=y_batch, mask=mask_batch)
            
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_steps

        # Conditional validation for speed (only every N epochs, use subset, skip early epochs)
        validation_frequency = EVALUATION_CONFIG.get('validation_frequency', 1)
        validation_sample_size = EVALUATION_CONFIG.get('validation_sample_size', None)
        eval_start_epoch = EVALUATION_CONFIG.get('eval_start_epoch', 1)
        
        should_evaluate = (epoch + 1) >= eval_start_epoch and (epoch + 1) % validation_frequency == 0
        
        if should_evaluate:
            # Use subset of validation data for faster evaluation
            if validation_sample_size and validation_sample_size < len(lines_val):
                val_lines_subset = lines_val[:validation_sample_size]
                val_Y_subset = Y_val[:validation_sample_size]
                print(f"  Using validation subset: {validation_sample_size}/{len(lines_val)} samples")
            else:
                val_lines_subset = lines_val
                val_Y_subset = Y_val
            
            # Validation (using competition mode for accurate metrics)
            val_accuracy, val_der = evaluate_model(
                model, val_lines_subset, val_Y_subset, vocab, embedder, device, diacritic2id, model_name, config
            )

            print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | DER: {val_der:.4f}")
            
            # Early stopping check
            if val_der < best_der:
                best_der = val_der
                patience_counter = 0
                print(f"  ✓ New best model! DER: {best_der:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                    # Save final checkpoint before stopping
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_der': best_der,
                        'config': config,
                        'vocab': vocab.char2id
                    }
                    torch.save(checkpoint, f"models/epoch_{epoch+1}_{model_name}.pth")
                    break
        else:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Validation skipped (every {validation_frequency} epochs)")
            val_der = best_der  # Keep best DER for comparison

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Save checkpoint EVERY epoch (not just best)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_der': best_der,
            'val_der': val_der,
            'train_loss': avg_train_loss,
            'config': config,
            'vocab': vocab.char2id
        }
        
        # Save with epoch number
        torch.save(checkpoint, f"models/epoch_{epoch+1}_{model_name}.pth")
        
        # Also update best model if this is the best so far
        if should_evaluate and val_der == best_der:
            torch.save(checkpoint, f"models/best_{model_name}.pth")
            print(f"  💾 Best model updated: models/best_{model_name}.pth")

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED!")
    print("="*70)
    print(f"Best DER: {best_der:.4f}")
    print(f"Model saved to: models/best_{model_name}.pth")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arabic Diacritization Models")
    parser.add_argument(
        "--model",
        choices=["rnn", "lstm", "crf", "bilstm_crf", "hierarchical_bilstm", "arabert_bilstm_crf", "arabert_char_bilstm_crf", "char_bilstm_classifier", "charngram_bilstm_classifier"],
        default="bilstm_crf",
        help="Model to train"
    )
    parser.add_argument(
        "--train_data",
        default="data/train.txt",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data",
        default="data/val.txt",
        help="Path to validation data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    train_model(args.model, args.train_data, args.val_data, args.max_samples, args.seed)