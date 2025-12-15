#!/usr/bin/env python3
"""
Arabic Diacritization Testing Script

Loads a trained model and evaluates it on test data
Supports multiple models: RNN, LSTM, CRF, BiLSTM-CRF
"""

import argparse
import sys
import os
import re
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Regex for removing diacritics
DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u0652]')

def remove_diacritics(text):
    """Remove all diacritics from Arabic text"""
    return DIACRITICS.sub('', text)

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
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in dataloader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)

            predictions = model(X_batch, mask=mask_batch)

            # predictions is a list of lists (one per sequence in batch)
            # y_batch is tensor, mask_batch is tensor
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
                if t != diacritic2id['']:
                    flat_predictions.append(p)
                    flat_targets.append(t)

    accuracy = accuracy_score(flat_targets, flat_predictions) if flat_targets else 0
    der = calculate_der(all_predictions, all_targets, all_masks)

    return accuracy, der

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
    else:
        raise ValueError(f"Model {model_name} not implemented yet")

    return model

def load_test_data(test_path, max_samples=None):
    """Load test data"""
    print("Loading test data...")
    X_test, Y_test, lines_test = tokenize_file(test_path)
    if max_samples:
        X_test = X_test[:max_samples]
        Y_test = Y_test[:max_samples]
        lines_test = lines_test[:max_samples]

    return X_test, Y_test, lines_test

def prepare_test_data(X, Y, lines, vocab, config, diacritic2id, embedder=None):
    """Prepare test data for evaluation"""
    if config.get("use_contextual", False):
        # Use custom dataset that computes embeddings on-the-fly
        print("Preparing test dataset (embeddings computed on-the-fly)...")
        dataset = ContextualDataset(X, Y, lines, vocab, config, diacritic2id, embedder)
    else:
        # Encode sequences
        X_encoded = [vocab.encode(seq) for seq in X]
        Y_encoded = encode_corpus(Y, diacritic2id)

        # Pad sequences
        X_padded = pad_sequences(X_encoded, config["max_seq_length"])
        Y_padded = pad_sequences(Y_encoded, config["max_seq_length"])
        masks = [[True] * len(seq) + [False] * (config["max_seq_length"] - len(seq)) for seq in X_encoded]

        dataset = TensorDataset(
            torch.tensor(X_padded, dtype=torch.long),
            torch.tensor(Y_padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool)
        )

    return dataset

def predict_diacritics(model, lines, vocab, config, diacritic2id, embedder, device, id2diacritic):
    """Predict diacritics for undiacritized lines"""
    model.eval()
    predictions = []

    for line in tqdm(lines, desc="Predicting"):
        # Remove diacritics from line
        undiacritized = remove_diacritics(line)

        # Tokenize undiacritized line to get base characters (without spaces)
        base_chars, _ = tokenize_line(undiacritized)

        if config.get("use_contextual", False):
            # Use embedder on undiacritized line
            emb = embedder.embed_line_chars(undiacritized)
            X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            mask = torch.tensor([True] * len(emb), dtype=torch.bool).unsqueeze(0).to(device)
        else:
            # Encode characters
            X_encoded = vocab.encode(base_chars)
            X_padded = pad_sequences([X_encoded], config["max_seq_length"])
            X_tensor = torch.tensor(X_padded, dtype=torch.long).to(device)
            mask = torch.tensor([[True] * len(X_encoded) + [False] * (config["max_seq_length"] - len(X_encoded))], dtype=torch.bool).to(device)

        with torch.no_grad():
            pred = model(X_tensor, mask=mask)

        # Convert predictions to diacritics
        pred_diacritics = []
        for p in pred[0][:len(base_chars)]:  # Only take predictions for actual characters
            pred_diacritics.append(id2diacritic.get(p, ''))

        # Reconstruct diacritized text by applying diacritics to the original undiacritized line
        diacritized = ""
        char_idx = 0
        for ch in undiacritized:
            if ch.isspace():
                diacritized += ch
            elif is_arabic_base_letter(ch) or is_arabic_digit(ch):
                if char_idx < len(pred_diacritics):
                    diacritized += ch + pred_diacritics[char_idx]
                    char_idx += 1
                else:
                    diacritized += ch
            else:
                diacritized += ch

        predictions.append(diacritized)

    return predictions

# Import our modules
from src.preprocessing.tokenize import tokenize_file, tokenize_line, is_arabic_base_letter, is_arabic_digit
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Arabic Diacritization Models")
    parser.add_argument(
        "--model",
        choices=["bilstm_crf"],  # Add more as implemented
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--test_data",
        default="data/val.txt",
        help="Path to test data (with diacritics for evaluation)"
    )
    parser.add_argument(
        "--predict_input",
        default="test_output.txt",
        help="Path to undiacritized text for prediction"
    )
    parser.add_argument(
        "--output_file",
        default="test_predictions.txt",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples"
    )

    args = parser.parse_args()

    # Set seed
    set_seed()

    # Get configuration
    config = get_model_config(args.model)
    if config is None:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Testing {args.model.upper()} model")
    print(f"Model path: {args.model_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load diacritic mapping
    with open("utils/diacritic2id.pickle", "rb") as f:
        diacritic2id = pickle.load(f)
    id2diacritic = {v: k for k, v in diacritic2id.items()}
    print(f"✓ Loaded {len(diacritic2id)} diacritic classes")

    # Initialize embedder if using contextual embeddings
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

    # Load model checkpoint
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    vocab_dict = checkpoint['vocab']

    # Rebuild vocab
    vocab = CharVocab()
    vocab.char2id = vocab_dict
    vocab.id2char = {v: k for k, v in vocab_dict.items()}

    # Update vocab size in config
    config = update_vocab_size(config.copy(), len(vocab.char2id))

    # Initialize model
    model = get_model(args.model, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✓ Model loaded successfully")

    # Load test data for evaluation
    X_test, Y_test, lines_test = load_test_data(args.test_data, args.max_samples)
    print(f"✓ Loaded {len(X_test)} test samples")

    # Prepare test dataset
    test_dataset = prepare_test_data(X_test, Y_test, lines_test, vocab, config, diacritic2id, embedder)

    # Create dataloader
    batch_size = 1 if config.get("use_contextual", False) else DATA_CONFIG['batch_size']
    collate_fn = collate_contextual_batch if config.get("use_contextual", False) else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_accuracy, test_der = evaluate_model(model, test_loader, device, diacritic2id)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test DER: {test_der:.4f}")

    # Load prediction input
    print(f"\nLoading prediction input from {args.predict_input}...")
    with open(args.predict_input, "r", encoding="utf-8") as f:
        predict_lines = [line.strip() for line in f if line.strip()]

    if args.max_samples:
        predict_lines = predict_lines[:args.max_samples]

    print(f"✓ Loaded {len(predict_lines)} lines for prediction")

    # Make predictions
    print("\nGenerating predictions...")
    predictions = predict_diacritics(model, predict_lines, vocab, config, diacritic2id, embedder, device, id2diacritic)

    # Save predictions
    print(f"\nSaving predictions to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print("✓ Predictions saved!")
    print("\n" + "="*70)
    print("TESTING COMPLETED!")
    print("="*70)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test DER: {test_der:.4f}")
    print(f"Predictions saved to: {args.output_file}")
    print("="*70)