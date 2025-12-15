#!/usr/bin/env python3
"""
Hierarchical BiLSTM Model for Arabic Diacritization

Combines word-level and character-level BiLSTM encoders
Adapted from reference implementation to work with our preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordLevelBiLSTM(nn.Module):
    """Word-level BiLSTM encoder"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Since bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, word_ids):
        # word_ids: (batch, seq_len)
        emb = self.embedding(word_ids)  # (batch, seq_len, embedding_dim)
        out, _ = self.bilstm(emb)  # (batch, seq_len, hidden_dim)
        return out


class CharLevelBiLSTM(nn.Module):
    """Character-level BiLSTM encoder"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Since bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, char_ids):
        # char_ids: (batch, seq_len)
        emb = self.embedding(char_ids)  # (batch, seq_len, embedding_dim)
        out, _ = self.bilstm(emb)  # (batch, seq_len, hidden_dim)
        return out


class HierarchicalBiLSTM(nn.Module):
    """
    Hierarchical BiLSTM model adapted for character-level diacritization

    Architecture:
    1. Character-level BiLSTM processes character sequences
    2. Word-level features are derived by pooling character features
    3. Combined features fed to classifier
    """
    def __init__(self, config):
        super().__init__()

        # Character-level components
        self.char_encoder = CharLevelBiLSTM(
            vocab_size=config["char_vocab_size"],
            embedding_dim=config["char_embedding_dim"],
            hidden_dim=config["char_hidden_dim"],
            num_layers=config["char_num_layers"],
            dropout=config["dropout"]
        )

        # For hierarchical processing, we'll use convolution to create word-level features
        self.word_conv = nn.Conv1d(
            config["char_hidden_dim"],
            config["word_hidden_dim"],
            kernel_size=3,
            padding=1
        )

        # Combined feature dimension
        combined_dim = config["char_hidden_dim"] + config["word_hidden_dim"]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config["classifier_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["classifier_hidden_dim"], config["num_classes"])
        )

    def forward(self, x, tags=None, mask=None):
        """
        Forward pass matching BiLSTMCRF interface

        Args:
            x: Input (embeddings for contextual, char IDs otherwise)
            tags: Target labels for training
            mask: Attention mask

        Returns:
            Loss for training, predictions for inference
        """
        if isinstance(x, torch.Tensor) and x.dtype == torch.long:
            # Character IDs
            char_features = self.char_encoder(x)  # (batch, seq_len, char_hidden)
        else:
            # Contextual embeddings
            char_features = x  # Already embeddings

        # Create word-level features using convolution
        # Transpose for conv1d: (batch, hidden, seq_len)
        char_features_t = char_features.transpose(1, 2)
        word_features = self.word_conv(char_features_t)  # (batch, word_hidden, seq_len)
        word_features = word_features.transpose(1, 2)  # (batch, seq_len, word_hidden)

        # Combine features
        combined = torch.cat([char_features, word_features], dim=-1)  # (batch, seq_len, combined_dim)

        # Apply classifier
        logits = self.classifier(combined)  # (batch, seq_len, num_classes)

        if tags is not None:
            # Training mode - compute loss
            # Simple cross-entropy loss (no CRF)
            loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
            # Flatten
            logits_flat = logits.view(-1, logits.size(-1))
            tags_flat = tags.view(-1)
            if mask is not None:
                mask_flat = mask.view(-1)
                loss = loss_fct(logits_flat[mask_flat], tags_flat[mask_flat])
            else:
                loss = loss_fct(logits_flat, tags_flat)
            return loss
        else:
            # Inference mode - return predictions
            predictions = torch.argmax(logits, dim=-1)
            return predictions