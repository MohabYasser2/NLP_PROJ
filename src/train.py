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
    """Custom dataset that computes embeddings on-the-fly to save memory"""
    def __init__(self, X, Y, lines, vocab, config, diacritic2id, embedder):
        self.X = X
        self.Y = Y
        self.lines = lines
        self.vocab = vocab
        self.config = config
        self.diacritic2id = diacritic2id
        self.embedder = embedder
        
        # Pre-encode labels to avoid redundant computation
        self.Y_encoded = encode_corpus(Y, diacritic2id)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        y_seq = self.Y_encoded[idx]
        
        # Compute embedding on-the-fly
        emb = self.embedder.embed_line_chars(line)
        
        # Pad to max length in batch (handled by collate_fn)
        y_padded = np.pad(y_seq, (0, max(0, len(emb) - len(y_seq))), mode='constant')
        
        return {
            'embedding': torch.tensor(emb, dtype=torch.float32),
            'label': torch.tensor(y_padded[:len(emb)], dtype=torch.long),
            'mask': torch.tensor([True] * len(emb), dtype=torch.bool)
        }


def collate_contextual_batch(batch):
    """Collate function to pad embeddings in a batch"""
    embeddings = [item['embedding'] for item in batch]
    labels = [item['label'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    # Find max length in this batch
    max_len = max(len(emb) for emb in embeddings)
    
    # Pad all to max_len
    padded_embeddings = []
    padded_labels = []
    padded_masks = []
    
    for emb, label, mask in zip(embeddings, labels, masks):
        pad_len = max_len - len(emb)
        if pad_len > 0:
            padded_emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len), value=0.0)
            padded_label = torch.nn.functional.pad(label.unsqueeze(0), (0, pad_len), value=0).squeeze(0)
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_emb = emb
            padded_label = label
            padded_mask = mask
        
        padded_embeddings.append(padded_emb)
        padded_labels.append(padded_label)
        padded_masks.append(padded_mask)
    
    return (
        torch.stack(padded_embeddings),
        torch.stack(padded_labels),
        torch.stack(padded_masks)
    )

# Import our modules
from src.preprocessing.tokenize import tokenize_file
from src.preprocessing.encode_labels import encode_corpus
from utils.vocab import CharVocab
from src.config import (
    get_model_config, update_vocab_size,
    DATA_CONFIG, TRAINING_CONFIG, EVALUATION_CONFIG
)
from src.preprocessing.pad_sequences import pad_sequences
from src.features.contextual_embeddings import ContextualEmbedder

# Import models
from src.models.bilstm_crf import BiLSTMCRF
from src.models.bilstm_classifier import BiLSTMClassifier, BiLSTMCRFClassifier
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


def prepare_data(X, Y, lines, vocab, config, diacritic2id, embedder=None):
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

        # Create dataset
        dataset = TensorDataset(X_padded, Y_padded, mask)

    return dataset


def get_model(model_name, config):
    """Initialize the specified model"""
    model_name = model_name.lower()
    
    if model_name == "bilstm_crf":
        # Original BiLSTM-CRF model
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "use_contextual": config.get("use_contextual", False)
        }
        model = BiLSTMCRF(**model_config)
    
    elif model_name == "bilstm_classifier":
        # Hierarchical BiLSTM (Word + Char level) without CRF
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "char_emb_dim": config.get("char_emb_dim", 128),
            "char_hidden_dim": config.get("char_hidden_dim", 128),
            "word_hidden_dim": config.get("word_hidden_dim", 128),
            "dropout": config.get("dropout", 0.3),
            "use_contextual": config.get("use_contextual", False)
        }
        model = BiLSTMClassifier(**model_config)
    
    elif model_name == "bilstm_classifier_crf":
        # Hierarchical BiLSTM (Word + Char level) with CRF
        model_config = {
            "vocab_size": config["vocab_size"],
            "tagset_size": config["tagset_size"],
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "char_emb_dim": config.get("char_emb_dim", 128),
            "char_hidden_dim": config.get("char_hidden_dim", 128),
            "word_hidden_dim": config.get("word_hidden_dim", 128),
            "dropout": config.get("dropout", 0.3),
            "use_contextual": config.get("use_contextual", False),
            "use_crf": True
        }
        model = BiLSTMCRFClassifier(**model_config)
    
    else:
        raise ValueError(f"Model '{model_name}' not implemented. Available: bilstm_crf, bilstm_classifier, bilstm_classifier_crf")

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


def evaluate_model(model, dataloader, device, diacritic2id):
    """Evaluate model on validation set"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in dataloader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)

            predictions = model(X_batch, mask=mask_batch)

            # Check if predictions are logits (from BiLSTMClassifier) or viterbi paths (from CRF)
            if isinstance(predictions, torch.Tensor):
                # BiLSTMClassifier returns logits: (batch, seq, num_classes)
                # Convert to class indices
                predictions = predictions.argmax(dim=-1)  # (batch, seq)
                
                # Iterate through batch
                for pred_seq, target_seq, mask_seq in zip(predictions, y_batch, mask_batch):
                    pred_flat = []
                    target_flat = []
                    
                    for p, t, m in zip(pred_seq, target_seq, mask_seq):
                        if m:
                            pred_flat.append(p.item())
                            target_flat.append(t.item())
                    
                    if pred_flat:
                        all_predictions.append(pred_flat)
                        all_targets.append(target_flat)
                        all_masks.append([True] * len(pred_flat))
            else:
                # CRF returns list of lists (one per sequence in batch)
                for pred_seq, target_seq, mask_seq in zip(predictions, y_batch, mask_batch):
                    pred_flat = []
                    target_flat = []
                    mask_flat = []

                    for p, t, m in zip(pred_seq, target_seq, mask_seq):
                        if m:
                            pred_flat.append(p)
                            target_flat.append(t.item())
                            mask_flat.append(True)

                    if pred_flat:
                        all_predictions.append(pred_flat)
                        all_targets.append(target_flat)
                        all_masks.append(mask_flat)

    # Calculate metrics (exclude spaces from accuracy like DER does)
    flat_predictions = []
    flat_targets = []

    for pred_seq, target_seq, mask_seq in zip(all_predictions, all_targets, all_masks):
        for p, t, m in zip(pred_seq, target_seq, mask_seq):
            if m:  # Only non-padded positions
                # Skip spaces (empty diacritic) for accuracy calculation
                if t != diacritic2id.get('', 0):
                    flat_predictions.append(p)
                    flat_targets.append(t)

    accuracy = accuracy_score(flat_targets, flat_predictions) if flat_targets else 0
    der = calculate_der(all_predictions, all_targets, all_masks)

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
            cache_dir=None  # Disable caching to avoid disk space issues on Kaggle
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

    # Prepare datasets
    train_dataset = prepare_data(X_train, Y_train, lines_train, vocab, config, diacritic2id, embedder)
    val_dataset = prepare_data(X_val, Y_val, lines_val, vocab, config, diacritic2id, embedder)

    # Create dataloaders
    # Note: batch_size should be reasonable when using contextual embeddings
    batch_size = DATA_CONFIG['batch_size'] if not config.get("use_contextual", False) else 1
    
    # Use custom collate function for contextual embeddings
    collate_fn = collate_contextual_batch if config.get("use_contextual", False) else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
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

    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for X_batch, y_batch, mask_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()

            loss = model(X_batch, tags=y_batch, mask=mask_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_steps

        # Validation
        val_accuracy, val_der = evaluate_model(model, val_loader, device, diacritic2id)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | DER: {val_der:.4f}")

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Early stopping
        if val_der < best_der:
            best_der = val_der
            patience_counter = 0
            # Save complete checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_der': best_der,
                'config': config,
                'vocab': vocab.char2id
            }
            torch.save(checkpoint, f"models/best_{model_name}.pth")
            print(f"  ✓ New best model! DER: {best_der:.4f} (saved at epoch {epoch+1})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

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
        choices=["rnn", "lstm", "crf", "bilstm_crf"],
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