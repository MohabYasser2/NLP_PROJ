# -*- coding: utf-8 -*-
"""
Arabic Diacritization Models with Dual-Pathway BiLSTM
Compatible with the main training pipeline in train.py

Models:
- BiLSTMDualPathway: Char-level + Word-level BiLSTM with AraBERT
- BiLSTMDualPathwayCRF: Same with CRF layer for sequence constraints

Uses:
- Same preprocessing as other models (from src/preprocessing/)
- Same diacritic mapping (from utils/diacritic2id.pickle)
- AraBERT embeddings for both character and word representations
"""

import torch
import torch.nn as nn


class BiLSTMDualPathway(nn.Module):
    """
    Dual-pathway BiLSTM model for Arabic diacritization.
    
    Architecture:
    - Character-level BiLSTM: processes learned character embeddings (128-dim)
    - Word-level BiLSTM: processes AraBERT embeddings (768-dim)
    - Concatenation: combines both pathways
    - Classifier: outputs diacritic labels
    
    This matches the original 2bilstm_classifier.py but with AraBERT for words.
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=768,
        hidden_dim=256,
        char_emb_dim=128,
        char_hidden_dim=256,
        word_hidden_dim=256,
        dropout=0.3,
        use_contextual=False
    ):
        """
        Args:
            vocab_size: Size of character vocabulary (for learned embeddings)
            tagset_size: Number of diacritic classes (15)
            embedding_dim: Dimension of word embeddings (768 for AraBERT)
            hidden_dim: Not directly used, for compatibility
            char_emb_dim: Embedding dimension for character level (128)
            char_hidden_dim: Hidden dimension of character-level BiLSTM (256)
            word_hidden_dim: Hidden dimension of word-level BiLSTM (256)
            dropout: Dropout rate
            use_contextual: Should be True for this model
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.char_emb_dim = char_emb_dim
        self.use_contextual = use_contextual
        
        # Character embedding layer (learned, not AraBERT)
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        
        # Character-level BiLSTM pathway
        # Input: learned character embeddings (128-dim)
        self.char_bilstm = nn.LSTM(
            input_size=char_emb_dim,
            hidden_size=char_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0.0
        )
        
        # Word-level BiLSTM pathway
        # Input: AraBERT embeddings (768-dim)
        self.word_bilstm = nn.LSTM(
            input_size=embedding_dim,  # AraBERT: 768-dim
            hidden_size=word_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0.0
        )
        
        # Combine both pathways
        char_out_dim = char_hidden_dim * 2  # 512
        word_out_dim = word_hidden_dim * 2  # 512
        combined_dim = char_out_dim + word_out_dim  # 1024
        
        # Classifier head (from original)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, tagset_size)  # 512 -> 15
        )
    
    def forward(self, char_ids, word_embeddings, mask=None):
        """
        Forward pass
        
        Args:
            char_ids: Character IDs (batch, seq_len) - for learned embeddings
            word_embeddings: AraBERT embeddings (batch, seq_len, 768)
            mask: Optional mask tensor for padding
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, tagset_size)
        """
        
        # Character-level pathway: embed char IDs and process
        char_emb = self.char_embedding(char_ids)  # (batch, seq_len, 128)
        char_out, _ = self.char_bilstm(char_emb)  # (batch, seq_len, 512)
        
        # Word-level pathway: process AraBERT embeddings
        word_out, _ = self.word_bilstm(word_embeddings)  # (batch, seq_len, 512)
        
        # Combine both pathways
        combined = torch.cat([char_out, word_out], dim=-1)  # (batch, seq_len, 1024)
        
        # Classifier head
        logits = self.classifier(combined)  # (batch, seq_len, tagset_size)
        
        return logits


class BiLSTMDualPathwayCRF(nn.Module):
    """
    Dual-pathway BiLSTM + CRF model for Arabic diacritization using AraBERT.
    
    Extends BiLSTMDualPathway with CRF layer for sequence labeling constraints.
    Better accuracy for structured prediction tasks.
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=768,
        hidden_dim=256,
        char_hidden_dim=256,
        word_hidden_dim=256,
        dropout=0.3,
        use_contextual=False,
        use_crf=True
    ):
        super().__init__()
        
        # Dual-pathway BiLSTM component
        self.bilstm = BiLSTMDualPathway(
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            char_hidden_dim=char_hidden_dim,
            word_hidden_dim=word_hidden_dim,
            dropout=dropout,
            use_contextual=use_contextual
        )
        
        # CRF layer
        self.use_crf = use_crf
        if use_crf:
            try:
                from torchcrf import CRF
            except ImportError:
                raise ImportError("Please install TorchCRF: pip install TorchCRF")
            self.crf = CRF(tagset_size, batch_first=True)
    
    def forward(self, X, tags=None, mask=None):
        """
        Forward pass
        
        Args:
            X: Tuple of (char_ids, word_embeddings) OR dict with 'char_ids' and 'word_embedding' keys
            tags: Optional target labels for CRF loss computation
            mask: Optional mask tensor for padding
        
        Returns:
            If tags is provided: CRF loss
            If tags is None: CRF viterbi path predictions
        """
        
        # Handle both forward methods (from DataLoader or direct call)
        if isinstance(X, (tuple, list)):
            char_ids, word_embeddings = X[0], X[1]
        else:
            # Assume it's already unpacked in train loop
            char_ids, word_embeddings = X, None
        
        # Get BiLSTM logits
        logits = self.bilstm(char_ids, word_embeddings, mask=mask)
        
        if not self.use_crf:
            return logits
        
        # CRF layer
        if tags is not None:
            # Training: compute loss
            if mask is not None:
                loss = -self.crf(logits, tags, mask=mask, reduction='mean')
            else:
                loss = -self.crf(logits, tags, reduction='mean')
            return loss
        else:
            # Inference: decode
            if mask is not None:
                predictions = self.crf.decode(logits, mask=mask)
            else:
                predictions = self.crf.decode(logits)
            return predictions
