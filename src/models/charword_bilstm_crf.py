"""
Character + Word BiLSTM-CRF Model

Improved model using both character and word-level features.
Demonstrates the impact of word-level context on diacritization.

Features:
- Character embeddings (trainable, 128-dim)
- Word embeddings (trainable, 128-dim)
- Word boundary awareness

Expected Performance: 90-94% accuracy
Training Time: 3-4 hours on CPU
"""

import torch
import torch.nn as nn
from TorchCRF import CRF


class CharWordBiLSTMCRF(nn.Module):
    """
    Character + Word context BiLSTM-CRF model.
    
    Architecture:
        Char IDs → Char Embedding (128) ─┐
                                          ├─ Concat (256) → BiLSTM (2 layers, 256) → Linear → CRF
        Word IDs → Word Embedding (128) ─┘
    
    Each character is associated with its word, providing word-level context.
    
    Args:
        char_vocab_size: Size of character vocabulary
        word_vocab_size: Size of word vocabulary
        tagset_size: Number of diacritic classes (15)
        char_embedding_dim: Character embedding dimension (128)
        word_embedding_dim: Word embedding dimension (128)
        hidden_dim: BiLSTM hidden dimension (256)
        num_layers: Number of BiLSTM layers (2)
        dropout: Dropout rate (0.3)
    """
    
    def __init__(
        self,
        char_vocab_size,
        word_vocab_size,
        tagset_size,
        char_embedding_dim=128,
        word_embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(
            char_vocab_size,
            char_embedding_dim,
            padding_idx=0
        )
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(
            word_vocab_size,
            word_embedding_dim,
            padding_idx=0
        )
        
        # Fusion dimension
        fusion_dim = char_embedding_dim + word_embedding_dim  # 256
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            fusion_dim,
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
    
    def forward(self, char_ids, word_ids, tags=None, mask=None):
        """
        Forward pass with dual inputs
        
        Args:
            char_ids: Character IDs (batch, seq_len)
            word_ids: Word IDs for each character position (batch, seq_len)
            tags: Target tags (batch, seq_len) - only for training
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Training: Loss scalar
            Inference: List of predictions
        """
        # Embed characters and words
        char_emb = self.char_embedding(char_ids)  # (batch, seq, 128)
        word_emb = self.word_embedding(word_ids)  # (batch, seq, 128)
        
        # Concatenate features (fusion)
        fused = torch.cat([char_emb, word_emb], dim=-1)  # (batch, seq, 256)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(fused)  # (batch, seq, 256)
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
