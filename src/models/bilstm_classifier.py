# -*- coding: utf-8 -*-
"""
Arabic Diacritization Model with Bi-LSTM (Hierarchical - Word + Char Level)
Compatible with the main training pipeline in train.py

Uses:
- Separate Character-level BiLSTM encoder
- Separate Word-level BiLSTM encoder  
- Concatenation of both embeddings
- Same preprocessing as original file
- Same diacritic mapping (from utils/diacritic2id.pickle)

Architecture:
- Character pathway: processes character-level embeddings
- Word pathway: derives word-level context from character outputs
- Concatenates both pathways
- Classifier head outputs diacritics

Note: This version is adapted for train.py pipeline compatibility.
For exact original architecture, use 2bilstm_classifier.py standalone.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Hierarchical BiLSTM model for Arabic diacritization
    - Character-level BiLSTM: processes character embeddings
    - Word-level BiLSTM: processes word-level context (from character outputs)
    - Both embeddings concatenated for final classification
    
    This is adapted from 2bilstm_classifier.py architecture to work with train.py
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=768,
        hidden_dim=256,
        char_emb_dim=128,
        char_hidden_dim=128,
        word_hidden_dim=128,
        dropout=0.3,
        use_contextual=False
    ):
        """
        Args:
            vocab_size: Size of character vocabulary
            tagset_size: Number of diacritic classes
            embedding_dim: Dimension of input embeddings (768 for AraBERT, or char embedding size)
            hidden_dim: Hidden dimension (used for word pathway)
            char_emb_dim: Character embedding dimension (if not using contextual)
            char_hidden_dim: Character BiLSTM hidden dimension
            word_hidden_dim: Word BiLSTM hidden dimension
            dropout: Dropout rate
            use_contextual: Whether using contextual embeddings (AraBERT)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_contextual = use_contextual
        
        # Input embedding layer (only if not using contextual embeddings)
        if not use_contextual:
            self.embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
            input_dim = char_emb_dim
        else:
            # When using contextual embeddings (AraBERT), input_dim is embedding_dim (768)
            input_dim = embedding_dim
        
        # ========================
        # CHARACTER-LEVEL PATHWAY
        # ========================
        # Processes character embeddings
        self.char_bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=char_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        char_output_dim = char_hidden_dim * 2  # 256 with bidirectional
        
        # ========================
        # WORD-LEVEL PATHWAY
        # ========================
        # Processes character outputs to derive word-level context
        self.word_bilstm = nn.LSTM(
            input_size=char_output_dim,
            hidden_size=word_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        word_output_dim = word_hidden_dim * 2  # 256 with bidirectional
        
        # ========================
        # CONCATENATION & CLASSIFICATION
        # ========================
        # Combined dimension: character + word embeddings
        combined_dim = char_output_dim + word_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, tagset_size)
        )
    
    def forward(self, X, mask=None, tags=None):
        """
        Forward pass
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, embedding_dim) for contextual
               or (batch_size, seq_len) for non-contextual
            mask: Optional mask tensor of shape (batch_size, seq_len) for padding
            tags: Optional target labels for computing loss (for training)
        
        Returns:
            If tags is provided: Scalar loss
            If tags is None: logits of shape (batch_size, seq_len, tagset_size)
        """
        
        # Handle both contextual and non-contextual inputs
        if not self.use_contextual:
            # X is token IDs, need to embed
            if X.dim() == 2:
                X = self.embedding(X)  # (batch, seq_len, char_emb_dim)
        
        batch_size, seq_len, _ = X.shape
        
        # ========================
        # CHARACTER-LEVEL PATHWAY
        # ========================
        # Process embeddings through character-level BiLSTM
        # Input: (batch, seq_len, input_dim)
        # Output: (batch, seq_len, char_output_dim)
        char_out, _ = self.char_bilstm(X)
        
        # ========================
        # WORD-LEVEL PATHWAY
        # ========================
        # Derive word-level context from character outputs
        # Input: (batch, seq_len, char_output_dim)
        # Output: (batch, seq_len, word_output_dim)
        word_out, _ = self.word_bilstm(char_out)
        
        # ========================
        # CONCATENATION
        # ========================
        # Concatenate character and word embeddings
        # char_out: (batch, seq_len, char_output_dim)
        # word_out: (batch, seq_len, word_output_dim)
        # combined: (batch, seq_len, combined_dim)
        combined = torch.cat([char_out, word_out], dim=-1)
        
        # ========================
        # CLASSIFICATION
        # ========================
        # Pass concatenated embeddings through classifier
        logits = self.classifier(combined)  # (batch, seq_len, tagset_size)
        
        # If tags provided, compute loss (for training)
        if tags is not None:
            # Flatten for loss computation
            logits_flat = logits.view(-1, self.tagset_size)
            tags_flat = tags.view(-1)
            
            # Create mask for padding
            if mask is not None:
                mask_flat = mask.view(-1).float()
            else:
                mask_flat = torch.ones(tags_flat.shape[0], device=tags_flat.device)
            
            # CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fn(logits_flat, tags_flat)
            loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
            
            return loss
        else:
            # Return logits for inference
            return logits


class BiLSTMCRFClassifier(nn.Module):
    """
    Hierarchical BiLSTM + CRF model for Arabic diacritization
    Extends BiLSTMClassifier with CRF layer for sequence labeling constraints
    """
    
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=768,
        hidden_dim=256,
        char_emb_dim=128,
        char_hidden_dim=128,
        dropout=0.3,
        use_contextual=False,
        use_crf=True
    ):
        super().__init__()
        
        # BiLSTM component
        self.bilstm = BiLSTMClassifier(
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
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
            X: Input tensor
            tags: Optional target labels for CRF loss computation
            mask: Optional mask tensor for padding
        
        Returns:
            If tags is provided: CRF loss
            If tags is None: CRF viterbi path predictions
        """
        
        # Get BiLSTM logits
        logits = self.bilstm(X, mask=mask)  # (batch, seq_len, tagset_size)
        
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
