"""
Simple Character BiLSTM Classifier (Baseline)

Architecture: Character embeddings → BiLSTM → Linear classifier
No CRF, no word features, no n-grams - just pure character-level modeling.

Features:
- Character embeddings only (128-dim)

Expected Performance: 82-87% accuracy
Training Time: 1-2 hours on CPU
"""

import torch
import torch.nn as nn


class CharBiLSTMClassifier(nn.Module):
    """
    Simple character-only BiLSTM classifier.
    
    Architecture:
        Char IDs → Char Embedding (128) → BiLSTM (2 layers, 256) → Dropout → Linear → Softmax
    
    Args:
        vocab_size: Character vocabulary size
        tagset_size: Number of diacritic classes (15)
        embedding_dim: Character embedding dimension (128)
        hidden_dim: BiLSTM hidden dimension (256)
        num_layers: Number of BiLSTM layers (2)
        dropout: Dropout rate (0.5)
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ):
        super().__init__()
        
        # Character embedding
        self.char_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Bidirectional doubles this
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output classifier
        self.fc = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, char_ids, tags=None, mask=None):
        """
        Forward pass
        
        Args:
            char_ids: Character IDs (batch, seq_len)
            tags: Target tags (batch, seq_len) - only for training
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Training: (logits, loss)
            Inference: (logits, predictions)
        """
        # Embed characters
        char_emb = self.char_embedding(char_ids)  # (batch, seq, 128)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(char_emb)  # (batch, seq, 256)
        lstm_out = self.dropout(lstm_out)
        
        # Classify
        logits = self.fc(lstm_out)  # (batch, seq, tagset_size)
        
        if tags is not None:
            # Training: compute cross-entropy loss
            # Flatten for loss computation
            logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, tagset_size)
            tags_flat = tags.view(-1)  # (batch*seq)
            
            # Cross-entropy loss (ignores padding automatically if mask applied properly)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fn(logits_flat, tags_flat)
            
            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.view(-1).float()
                loss_per_token = loss_per_token * mask_flat
                loss = loss_per_token.sum() / mask_flat.sum()
            else:
                loss = loss_per_token.mean()
            
            return logits, loss
        else:
            # Inference: return predictions
            predictions = torch.argmax(logits, dim=-1)  # (batch, seq)
            return logits, predictions
