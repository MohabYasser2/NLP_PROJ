"""
Character-level BiLSTM-CRF Model

Simple baseline model using only character embeddings.
No word-level or contextual features.

Features:
- Character embeddings (trainable, 128-dim)

Expected Performance: 85-90% accuracy
Training Time: 2-3 hours on CPU
"""

import torch
import torch.nn as nn
from TorchCRF import CRF


class CharBiLSTMCRF(nn.Module):
    """
    Character-only BiLSTM-CRF baseline model.
    
    Architecture:
        Char IDs → Char Embedding (128) → BiLSTM (2 layers, 256) → Linear → CRF
    
    Args:
        vocab_size: Size of character vocabulary
        tagset_size: Number of diacritic classes (15)
        embedding_dim: Character embedding dimension (128)
        hidden_dim: BiLSTM hidden dimension (256)
        num_layers: Number of BiLSTM layers (2)
        dropout: Dropout rate (0.3)
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # Character embedding layer
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
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size)
    
    def forward(self, char_ids, tags=None, mask=None):
        """
        Forward pass
        
        Args:
            char_ids: Character IDs (batch, seq_len)
            tags: Target tags (batch, seq_len) - only for training
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Training: Loss scalar
            Inference: List of predictions
        """
        # Embed characters
        char_emb = self.char_embedding(char_ids)  # (batch, seq, 128)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(char_emb)  # (batch, seq, 256)
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        emissions = self.fc(lstm_out)  # (batch, seq, tagset_size)
        
        if tags is not None:
            # Training: CRF loss
            # TorchCRF expects (seq, batch, tags)
            emissions_t = emissions.transpose(0, 1)
            tags_t = tags.transpose(0, 1)
            mask_t = mask.transpose(0, 1) if mask is not None else None
            
            loss = -self.crf(emissions_t, tags_t, mask=mask_t).sum()
            return loss
        else:
            # Inference: Viterbi decoding
            emissions_t = emissions.transpose(0, 1)
            mask_t = mask.transpose(0, 1) if mask is not None else None
            
            predictions = self.crf.viterbi_decode(emissions_t, mask=mask_t)
            return predictions
