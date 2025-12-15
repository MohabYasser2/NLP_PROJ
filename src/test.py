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
    """
    Custom dataset that computes embeddings on-the-fly to save memory.
    Returns both AraBERT embeddings AND character IDs for fusion models.
    """
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

        # 1) Compute AraBERT embedding on-the-fly (per-character)
        emb = self.embedder.embed_line_chars(line)  # (T, 768)
        
        # 2) Encode character IDs (for morphology fusion)
        chars = list(line)
        char_ids = self.vocab.encode(chars)  # (T,)
        
        # Align lengths safely
        T = min(len(emb), len(char_ids), len(y_seq))
        emb = emb[:T]
        char_ids = char_ids[:T]
        y_seq = y_seq[:T]

        return {
            'embedding': torch.tensor(emb, dtype=torch.float32),      # (T, 768)
            'char_ids': torch.tensor(char_ids, dtype=torch.long),     # (T,)
            'label': torch.tensor(y_seq, dtype=torch.long),           # (T,)
            'mask': torch.tensor([True] * T, dtype=torch.bool)        # (T,)
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

def evaluate_model(model, dataloader, device, diacritic2id, model_name="bilstm_crf"):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []
    
    # Check model type
    is_fusion_model = model_name.lower() == "arabert_char_bilstm_crf"
    is_ngram_classifier = model_name.lower() == "charngram_bilstm_classifier"
    is_simple_classifier = model_name.lower() == "char_bilstm_classifier"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            if is_fusion_model:
                # Fusion model: unpack 4 tensors (embedding, char_ids, labels, mask)
                X_batch, char_ids_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                char_ids_batch = char_ids_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                # Forward pass with dual inputs
                predictions = model(X_batch, char_ids_batch, mask=mask_batch)
            elif is_ngram_classifier:
                # N-gram classifier: unpack 4 tensors (char_ids, ngram_ids, labels, mask)
                char_ids_batch, ngram_ids_batch, y_batch, mask_batch = batch
                char_ids_batch = char_ids_batch.to(device)
                ngram_ids_batch = ngram_ids_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                # Forward pass returns (logits, predictions)
                _, predictions = model(char_ids_batch, ngram_ids_batch, mask=mask_batch)
            elif is_simple_classifier:
                # Simple classifier: unpack 3 tensors (char_ids, labels, mask)
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                # Forward pass returns (logits, predictions)
                _, predictions = model(X_batch, mask=mask_batch)
            else:
                # Standard CRF model: unpack 3 tensors (X, y, mask)
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                # Forward pass with single input
                predictions = model(X_batch, mask=mask_batch)

            # Handle different prediction formats
            if isinstance(predictions, list):
                # CRF models: predictions is a list of lists (one per sequence in batch)
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
            else:
                # Non-CRF models: predictions is tensor (batch, seq_len)
                predictions = predictions.cpu()
                for pred_seq, target_seq, mask_seq in zip(predictions, y_batch, mask_batch):
                    pred_flat = []
                    target_flat = []
                    mask_flat = []

                    for p, t, m in zip(pred_seq, target_seq, mask_seq):
                        if m:
                            pred_flat.append(p.item())
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

    accuracy = accuracy_score(flat_targets, [p.cpu().item() if isinstance(p, torch.Tensor) else p for p in flat_predictions]) if flat_targets else 0
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
        model = AraBERTBiLSTMCRF(**model_config)
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

def load_test_data(test_path, max_samples=None):
    """Load test data"""
    print("Loading test data...")
    X_test, Y_test, lines_test = tokenize_file(test_path)
    total_loaded = len(X_test)
    if max_samples:
        X_test = X_test[:max_samples]
        Y_test = Y_test[:max_samples]
        lines_test = lines_test[:max_samples]
        print(f"  (Limited from {total_loaded} to {len(X_test)} samples)")

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
        X_padded, mask = pad_sequences(X_encoded, pad_value=vocab.char2id["<PAD>"])
        Y_padded, _ = pad_sequences(Y_encoded, pad_value=0)  # Use 0 for padding (valid tag index)

        dataset = TensorDataset(
            X_padded,
            Y_padded,
            mask
        )

    return dataset

def predict_diacritics(model, lines, vocab, config, diacritic2id, embedder, device, id2diacritic):
    """Predict diacritics for undiacritized lines"""
    model.eval()
    predictions = []
    
    # Detect model type
    is_fusion_model = hasattr(model, 'char_embedding') and hasattr(model, 'arabert_projection')
    is_ngram_classifier = hasattr(model, 'ngram_embedding') and hasattr(model, 'char_embedding')
    is_simple_classifier = hasattr(model, 'char_embedding') and not hasattr(model, 'arabert_projection') and not hasattr(model, 'ngram_embedding')

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
            
            # Fusion model needs char_ids too
            if is_fusion_model:
                char_ids = vocab.encode(base_chars)
                char_ids_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
        else:
            # Encode characters
            X_encoded = vocab.encode(base_chars)
            X_padded, mask_tensor = pad_sequences([X_encoded], pad_value=vocab.char2id["<PAD>"])
            X_tensor = X_padded.to(device)
            mask = mask_tensor.to(device)

        with torch.no_grad():
            if config.get("use_contextual", False) and is_fusion_model:
                # Fusion model needs both embeddings and char_ids
                pred = model(X_tensor, char_ids_tensor, mask=mask)
            elif is_ngram_classifier:
                # N-gram classifier needs char_ids and ngram_ids
                char_ids = vocab.encode(base_chars)
                char_ids_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
                # For now, we'll use char_ids as ngram_ids (simplified)
                ngram_ids_tensor = char_ids_tensor.clone()
                _, pred = model(char_ids_tensor, ngram_ids_tensor, mask=mask)
            elif is_simple_classifier:
                # Simple classifier just needs char_ids
                _, pred = model(X_tensor, mask=mask)
            else:
                # Standard CRF model
                pred = model(X_tensor, mask=mask)

        # Convert predictions to diacritics
        if isinstance(pred, list):
            # CRF models return list of predictions
            pred_diacritics = [id2diacritic.get(p, '') for p in pred[0][:len(base_chars)]]
        else:
            # Classifier models return tensor
            pred_diacritics = [id2diacritic.get(p.item(), '') for p in pred[0][:len(base_chars)]]

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
from src.models.arabert_bilstm_crf import AraBERTBiLSTMCRF
from src.models.arabert_char_bilstm_crf import AraBERTCharBiLSTMCRF
from src.models.char_bilstm_classifier import CharBiLSTMClassifier
from src.models.charngram_bilstm_classifier import CharNgramBiLSTMClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Arabic Diacritization Models")
    parser.add_argument(
        "--model",
        choices=["bilstm_crf", "hierarchical_bilstm", "arabert_bilstm_crf", "arabert_char_bilstm_crf", "char_bilstm_classifier", "charngram_bilstm_classifier"],
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
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Competition mode: generate CSV with character IDs and diacritic labels"
    )
    parser.add_argument(
        "--competition_input",
        default=None,
        help="Path to undiacritized text file for competition (one line per sample)"
    )
    parser.add_argument(
        "--competition_output",
        default="competition_submission.csv",
        help="Path to save competition CSV output"
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
    
    # Add max_seq_length if not present (for hierarchical model)
    if "max_seq_length" not in config:
        config["max_seq_length"] = 256

    # Initialize model
    model = get_model(args.model, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✓ Model loaded successfully")

    # Competition mode: Generate CSV with character IDs and diacritic labels
    if args.competition:
        if not args.competition_input:
            raise ValueError("--competition_input is required in competition mode")
        
        print("\n" + "="*70)
        print("COMPETITION MODE")
        print("="*70)
        
        # Load undiacritized text
        print(f"\nLoading undiacritized text from {args.competition_input}...")
        with open(args.competition_input, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Loaded {len(lines)} lines")
        
        # Predict diacritics for all lines
        print("\nGenerating diacritic predictions...")
        model.eval()
        
        # Detect model type
        is_fusion_model = hasattr(model, 'char_embedding') and hasattr(model, 'arabert_projection')
        is_ngram_classifier = hasattr(model, 'ngram_embedding') and hasattr(model, 'char_embedding')
        is_simple_classifier = hasattr(model, 'char_embedding') and not hasattr(model, 'arabert_projection') and not hasattr(model, 'ngram_embedding')
        
        all_predictions = []
        char_id = 0
        
        for line_idx, line in enumerate(tqdm(lines, desc="Processing lines")):
            # Remove diacritics from line (if any)
            undiacritized = remove_diacritics(line)
            
            # Get base characters using tokenize_line (consistent with training)
            base_chars, _ = tokenize_line(undiacritized)
            
            if not base_chars:
                continue
            
            # Prepare input based on model type
            if config.get("use_contextual", False):
                # Use embedder on undiacritized line
                emb = embedder.embed_line_chars(undiacritized)
                
                # For fusion models, we need matching char_ids
                if is_fusion_model:
                    # Get char_ids that match the embedding length
                    # embed_line_chars returns embeddings for all characters in the processed line
                    # We need to ensure base_chars matches this
                    char_ids = vocab.encode(base_chars)
                    
                    # Align lengths: emb should match char_ids length
                    min_len = min(len(emb), len(char_ids))
                    emb = emb[:min_len]
                    char_ids = char_ids[:min_len]
                    
                    X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    char_ids_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
                    mask = torch.tensor([True] * min_len, dtype=torch.bool).unsqueeze(0).to(device)
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
            with torch.no_grad():
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
                    # Check if it's the expected format [[label1, label2, ...]] or wrong format [[label1], [label2], ...]
                    if len(pred) == 1:
                        # Correct format: one sequence in batch
                        pred_labels = pred[0][:len(base_chars)]
                    else:
                        # Wrong format from CRF: flatten the nested lists
                        pred_labels = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in pred]
                        pred_labels = pred_labels[:len(base_chars)]
                else:
                    # pred is already a flat list
                    pred_labels = pred[:len(base_chars)]
            else:
                pred_labels = pred[0][:len(base_chars)].cpu().tolist()
            
            # Store predictions with IDs
            for label in pred_labels:
                if isinstance(label, torch.Tensor):
                    label = label.item()
                all_predictions.append((char_id, int(label)))
                char_id += 1
        
        # Write to CSV
        print(f"\nSaving competition output to {args.competition_output}...")
        with open(args.competition_output, "w", encoding="utf-8") as f:
            f.write("ID,label\n")
            for char_id, label in all_predictions:
                f.write(f"{char_id},{label}\n")
        
        print("✓ Competition CSV saved!")
        print(f"  Total characters: {len(all_predictions)}")
        print("\n" + "="*70)
        print("COMPETITION MODE COMPLETED!")
        print("="*70)
        print(f"Output file: {args.competition_output}")
        print("="*70)
        
        # Exit after competition mode
        import sys
        sys.exit(0)

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
    test_accuracy, test_der = evaluate_model(model, test_loader, device, diacritic2id, args.model)
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