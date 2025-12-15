"""
AraBERT + Character Morphology Fusion BiLSTM-CRF Model

This model combines two complementary feature types:
1. AraBERT contextual embeddings (768-dim) - captures semantic/word-level context
2. Character embeddings (100-dim) - captures morphological/orthographic patterns

Architecture:
    AraBERT Embeddings (768) ──┐
                                ├─> Concatenate -> BiLSTM (2 layers) -> Linear -> CRF
    Char Embeddings (100) ─────┘

Key improvements over baseline bilstm_crf.py:
- Dual feature fusion (semantic + morphological)
- AraBERT projection with LayerNorm + Dropout for regularization
- 2-layer BiLSTM for deeper representation
- Dropout between layers for better generalization
"""

import torch
import torch.nn as nn
from TorchCRF import CRF  # type: ignore


class AraBERTCharBiLSTMCRF(nn.Module):
    """
    Fusion model: AraBERT contextual + Character morphology
    
    Args:
        char_vocab_size: Size of character vocabulary
        tagset_size: Number of diacritic classes (15)
        arabert_dim: AraBERT embedding dimension (768)
        char_embedding_dim: Character embedding dimension (100)
        hidden_dim: BiLSTM hidden dimension (512 recommended)
        num_layers: Number of BiLSTM layers (2 recommended)
        dropout: Dropout rate (0.3 recommended)
    """
    
    def __init__(
        self,
        char_vocab_size,
        tagset_size,
        arabert_dim=768,
        char_embedding_dim=100,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # Character embedding layer (for morphological patterns)
        self.char_embedding = nn.Embedding(
            char_vocab_size, 
            char_embedding_dim, 
            padding_idx=0
        )
        
        # AraBERT projection with normalization
        # Projects 768 -> smaller dim, adds regularization
        self.arabert_projection = nn.Sequential(
            nn.Linear(arabert_dim, arabert_dim // 2),  # 768 -> 384
            nn.LayerNorm(arabert_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer input size
        fusion_input_dim = (arabert_dim // 2) + char_embedding_dim  # 384 + 100 = 484
        
        # BiLSTM encoder (2 layers with dropout)
        self.lstm = nn.LSTM(
            fusion_input_dim,
            hidden_dim // 2,  # Divided by 2 because bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # Dropout between LSTM layers
        )
        
        # Output projection to diacritic classes
        self.fc = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer for sequence decoding
        self.crf = CRF(tagset_size)
        
        # Dropout after BiLSTM
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, arabert_emb, char_ids, tags=None, mask=None):
        """
        Forward pass with dual inputs
        
        Args:
            arabert_emb: AraBERT embeddings (batch, seq_len, 768)
            char_ids: Character IDs (batch, seq_len)
            tags: Target diacritic tags (batch, seq_len) - only for training
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Training mode (tags provided): Loss scalar
            Inference mode (tags=None): List of predictions
        """
        # 1. Project AraBERT embeddings
        arabert_proj = self.arabert_projection(arabert_emb)  # (batch, seq, 384)
        
        # 2. Embed character IDs
        char_emb = self.char_embedding(char_ids)  # (batch, seq, 100)
        
        # 3. Concatenate features (fusion)
        fused = torch.cat([arabert_proj, char_emb], dim=-1)  # (batch, seq, 484)
        
        # 4. BiLSTM encoding
        lstm_out, _ = self.lstm(fused)  # (batch, seq, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        
        # 5. Project to tag space
        emissions = self.fc(lstm_out)  # (batch, seq, tagset_size)
        
        # 6. CRF layer
        if tags is not None:
            # Training: compute CRF loss
            # TorchCRF expects (seq_len, batch, num_tags)
            emissions_t = emissions.transpose(0, 1)  # (seq, batch, tags)
            tags_t = tags.transpose(0, 1)  # (seq, batch)
            mask_t = mask.transpose(0, 1) if mask is not None else None  # (seq, batch)
            
            # CRF returns negative log-likelihood per sequence
            loss = -self.crf(emissions_t, tags_t, mask=mask_t).sum()
            return loss
        else:
            # Inference: Viterbi decoding
            emissions_t = emissions.transpose(0, 1)  # (seq, batch, tags)
            mask_t = mask.transpose(0, 1) if mask is not None else None  # (seq, batch)
            
            predictions = self.crf.viterbi_decode(emissions_t, mask=mask_t)
            return predictions
