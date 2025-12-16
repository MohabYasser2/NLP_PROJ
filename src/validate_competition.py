# -*- coding: utf-8 -*-
"""
Competition-style validation during training

Extracts diacritics from validation set, runs model in competition mode,
and calculates real accuracy to match competition evaluation.
"""

import sys
import os
import pickle
import tempfile
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.tokenize import tokenize_line
import re

# Diacritics pattern
DIACRITICS = re.compile(r'[\u064B-\u065F]')

def remove_diacritics(text):
    """Remove all diacritics from Arabic text"""
    return DIACRITICS.sub('', text)


def validate_competition_style(model, lines, vocab, config, diacritic2id, embedder, device, model_name):
    """
    Validate using competition-style evaluation.
    
    Args:
        model: Trained model
        lines: List of diacritized validation lines
        vocab: Character vocabulary
        config: Model configuration
        diacritic2id: Diacritic to ID mapping
        embedder: Contextual embedder (if using AraBERT)
        device: torch device
        model_name: Name of model
    
    Returns:
        accuracy: Character-level accuracy
        der: Diacritic Error Rate
    """
    from tqdm import tqdm
    
    model.eval()
    
    # Detect model type
    is_fusion_model = hasattr(model, 'char_embedding') and hasattr(model, 'arabert_projection')
    is_ngram_classifier = hasattr(model, 'ngram_embedding') and hasattr(model, 'char_embedding')
    is_simple_classifier = hasattr(model, 'char_embedding') and not hasattr(model, 'arabert_projection') and not hasattr(model, 'ngram_embedding')
    
    id2diacritic = {v: k for k, v in diacritic2id.items()}
    
    total_chars = 0
    correct_chars = 0
    
    with torch.no_grad():
        for line in tqdm(lines, desc="Validating (competition-style)", leave=False):
            # Extract golden labels
            base_chars, golden_diacritics = tokenize_line(line)
            
            if not base_chars:
                continue
            
            # Convert golden diacritics to labels
            golden_labels = []
            for diac in golden_diacritics:
                # Canonicalize diacritic
                if diac and 'ّ' in diac and len(diac) > 1:
                    diac_without_shadda = diac.replace('ّ', '')
                    diac = 'ّ' + diac_without_shadda
                
                if diac in diacritic2id:
                    golden_labels.append(diacritic2id[diac])
                else:
                    golden_labels.append(0)  # Unknown -> empty
            
            # Remove diacritics for prediction
            undiacritized = remove_diacritics(line)
            
            # Prepare input based on model type
            if config.get("use_contextual", False):
                # Use embedder on undiacritized line
                emb = embedder.embed_line_chars(undiacritized)
                
                # For fusion models, we need matching char_ids
                if is_fusion_model:
                    char_ids = vocab.encode(base_chars)
                    
                    # Align lengths
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
                from src.preprocessing.pad_sequences import pad_sequences
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
            
            # Compare with golden labels
            for idx, (pred_label, golden_label) in enumerate(zip(pred_labels, golden_labels)):
                if isinstance(pred_label, torch.Tensor):
                    pred_label = pred_label.item()
                
                total_chars += 1
                if int(pred_label) == int(golden_label):
                    correct_chars += 1
    
    accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    der = 1.0 - accuracy
    
    return accuracy, der
