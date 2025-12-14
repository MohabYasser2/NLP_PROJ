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
    Dual-pathway BiLSTM model for Arabic diacritization with word-level AraBERT.
    
    Architecture:
    - Character-level BiLSTM: processes learned character embeddings (128-dim)
    - Word-level BiLSTM: processes AraBERT embeddings (768-dim) at WORD level
    - Expansion: word pathway output expanded to character level (like original design)
    - Concatenation: combines both pathways at character level
    - Classifier: outputs diacritic labels
    
    Matches the original 2bilstm_classifier.py design but uses AraBERT for words.
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
        # Input: AraBERT embeddings (768-dim) at WORD level
        self.word_bilstm = nn.LSTM(
            input_size=embedding_dim,  # AraBERT: 768-dim
            hidden_size=word_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0.0
        )
        
        # Combine both pathways
        self.char_out_dim = char_hidden_dim * 2  # 512
        self.word_out_dim = word_hidden_dim * 2  # 512
        combined_dim = self.char_out_dim + self.word_out_dim  # 1024
        
        # Classifier head (from original)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, tagset_size)  # 512 -> 15
        )
    
    def forward(self, X, tags=None, mask=None):
        """
        Forward pass
        
        Args:
            X: Tuple of (char_ids, word_embeddings, word_boundaries)
               char_ids: Character IDs (batch, seq_len) - for learned embeddings
               word_embeddings: List of AraBERT embeddings [(num_words_0, 768), ...]
               word_boundaries: List of word boundaries per sample
            tags: Not used (for compatibility with train.py)
            mask: Optional mask tensor for padding
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, tagset_size)
        """
        
        # Handle tuple input
        if isinstance(X, (tuple, list)) and len(X) == 3:
            char_ids, word_embeddings, word_boundaries = X
        else:
            raise ValueError("X must be a tuple of (char_ids, word_embeddings, word_boundaries)")
        
        batch_size = char_ids.size(0)
        char_seq_len = char_ids.size(1)
        
        # CHARACTER PATHWAY: embed char IDs and process
        char_emb = self.char_embedding(char_ids)  # (batch, seq_len, 128)
        char_out, _ = self.char_bilstm(char_emb)  # (batch, seq_len, 512)
        
        # WORD PATHWAY: process word-level AraBERT embeddings
        # word_embeddings is a list of (num_words_i, 768) tensors
        # We need to process each sample in the batch separately because they have different word counts
        word_out_expanded_list = []
        
        for b in range(batch_size):
            word_emb = word_embeddings[b]  # (num_words, 768)
            word_boundary = word_boundaries[b]  # List of word lengths
            
            if len(word_emb) == 0:
                # No words in this sample (edge case)
                word_out_expanded = torch.zeros(
                    char_seq_len, self.word_out_dim, 
                    device=char_out.device, dtype=char_out.dtype
                )
            else:
                # Process words through word BiLSTM
                word_emb_unsqueezed = word_emb.unsqueeze(0)  # (1, num_words, 768)
                word_out_lstm, _ = self.word_bilstm(word_emb_unsqueezed)  # (1, num_words, 512)
                word_out_lstm = word_out_lstm.squeeze(0)  # (num_words, 512)
                
                # Expand word output to character level (like original .repeat() logic)
                # Each word embedding is repeated for each character in that word
                word_out_expanded = []
                for w_idx, (word_out_vec, num_chars) in enumerate(zip(word_out_lstm, word_boundary)):
                    # Repeat each word vector num_chars times
                    repeated = word_out_vec.unsqueeze(0).repeat(num_chars, 1)  # (num_chars, 512)
                    word_out_expanded.append(repeated)
                
                word_out_expanded = torch.cat(word_out_expanded, dim=0)  # (total_chars, 512)
                
                # Pad to match char_seq_len if needed (shouldn't be needed if data is aligned)
                if word_out_expanded.size(0) < char_seq_len:
                    pad_len = char_seq_len - word_out_expanded.size(0)
                    padding = torch.zeros(pad_len, self.word_out_dim, 
                                        device=word_out_expanded.device, dtype=word_out_expanded.dtype)
                    word_out_expanded = torch.cat([word_out_expanded, padding], dim=0)
                elif word_out_expanded.size(0) > char_seq_len:
                    word_out_expanded = word_out_expanded[:char_seq_len]
            
            word_out_expanded_list.append(word_out_expanded)
        
        # Stack all word outputs
        word_out_expanded = torch.stack(word_out_expanded_list, dim=0)  # (batch, seq_len, 512)
        
        # Concatenate both pathways at character level
        combined = torch.cat([char_out, word_out_expanded], dim=-1)  # (batch, seq_len, 1024)
        
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
            X: Tuple of (char_ids, word_embeddings, word_boundaries)
            tags: Optional target labels for CRF loss computation
            mask: Optional mask tensor for padding
        
        Returns:
            If tags is provided: CRF loss (or logits if CRF disabled)
            If tags is None: CRF viterbi path predictions (or logits if CRF disabled)
        """
        
        # Handle tuple input - pass all 3 components
        if isinstance(X, (tuple, list)) and len(X) >= 3:
            # X already has (char_ids, word_embeddings, word_boundaries)
            pass
        else:
            raise ValueError("X must be a tuple of (char_ids, word_embeddings, word_boundaries)")
        
        # Get BiLSTM logits
        logits = self.bilstm(X, tags=None, mask=mask)  # (batch, seq_len, tagset_size)
        
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
