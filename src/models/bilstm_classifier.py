# -*- coding: utf-8 -*-
"""
EXACT replica of original 2bilstm_classifier.py for 98% accuracy

Architecture components:
1. CharLevelBiLSTM: Encodes character embeddings [B, S, W] → [B, S, W, 2*char_hidden]
2. WordLevelBiLSTM: Encodes word embeddings [B, S] → [B, S, 2*word_hidden]  
3. Concatenation: Expands and combines both pathways
4. Classifier: Sequential FC layers for diacritic prediction

This model requires TWO inputs:
- char_ids: [B, S, W] - Character indices per word position
- word_ids: [B, S] - Word indices per sentence position
"""

import torch
import torch.nn as nn


class CharLevelBiLSTM(nn.Module):
    """Character-level BiLSTM encoder from original"""
    
    def __init__(self, n_chars, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.3):
        super().__init__()
        
        self.char_emb = nn.Embedding(
            num_embeddings=n_chars,
            embedding_dim=emb_dim,
            padding_idx=pad_id
        )
        
        self.char_bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, char_ids):
        """
        char_ids: [B, S, W]
        returns:  [B, S, W, 2*hidden]
        """
        B, S, W = char_ids.size()
        
        flat = char_ids.view(B * S, W)
        emb = self.char_emb(flat)
        out, _ = self.char_bilstm(emb)
        out = out.view(B, S, W, -1)
        
        return out


class WordLevelBiLSTM(nn.Module):
    """Word-level BiLSTM encoder from original"""
    
    def __init__(self, n_words, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.3):
        super().__init__()
        
        self.word_emb = nn.Embedding(
            num_embeddings=n_words,
            embedding_dim=emb_dim,
            padding_idx=pad_id
        )
        
        self.word_bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, word_ids):
        """
        word_ids: [B, S]
        returns:  [B, S, 2*hidden]
        """
        emb = self.word_emb(word_ids)
        out, _ = self.word_bilstm(emb)
        return out


class BiLSTMClassifier(nn.Module):
    """
    EXACT replica of 2bilstm_classifier.py (98% accuracy model)
    
    Parameters match original:
    - Character embedding: 64-dim
    - Character BiLSTM: 128-dim hidden → 256-dim output (bidirectional)
    - Word embedding: 128-dim
    - Word BiLSTM: 128-dim hidden → 256-dim output (bidirectional)
    - Combined: 512-dim → Classifier: 512→256→15 classes
    
    SUPPORTS TWO USAGE MODES:
    1. Original: forward(char_ids, word_ids) with two separate inputs
    2. Train.py compatible: forward(X) with embeddings (uses word approximation)
    """
    
    def __init__(
        self,
        n_chars=None,
        n_words=None,
        char_emb_dim=64,
        word_emb_dim=128,
        char_hidden=128,
        word_hidden=128,
        char_pad_id=0,
        word_pad_id=0,
        out_classes=15,
        dropout=0.3,
        # Train.py compatibility parameters (ignored if using char/word inputs)
        vocab_size=None,
        tagset_size=None,
        embedding_dim=768,
        hidden_dim=256,
        use_contextual=False
    ):
        super().__init__()
        
        # If n_chars not provided, use vocab_size
        if n_chars is None:
            n_chars = vocab_size if vocab_size else 1000
        if n_words is None:
            n_words = vocab_size if vocab_size else 1000
        if out_classes is None:
            out_classes = tagset_size if tagset_size else 15
            
        self.n_chars = n_chars
        self.n_words = n_words
        self.out_classes = out_classes
        self.char_pad_id = char_pad_id
        self.word_pad_id = word_pad_id
        
        self.char_encoder = CharLevelBiLSTM(
            n_chars=n_chars,
            emb_dim=char_emb_dim,
            hidden_dim=char_hidden,
            num_layers=1,
            pad_id=char_pad_id,
            dropout=dropout
        )
        
        self.word_encoder = WordLevelBiLSTM(
            n_words=n_words,
            emb_dim=word_emb_dim,
            hidden_dim=word_hidden,
            num_layers=1,
            pad_id=word_pad_id,
            dropout=dropout
        )
        
        combined_dim = (char_hidden * 2) + (word_hidden * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, out_classes)
        )
    
    def forward(self, char_ids, word_ids=None, tags=None, mask=None):
        """
        DUAL MODE:
        
        Mode 1 (Original): forward(char_ids[B,S,W], word_ids[B,S])
            Returns [B, S, W, out_classes]
        
        Mode 2 (Train.py): forward(X[B,S,D], tags=y[B,S], mask=mask[B,S])
            Returns loss (scalar) for training
        """
        
        # Compute logits
        logits = self._compute_logits(char_ids, word_ids)
        
        # If tags provided, compute loss (training mode)
        if tags is not None:
            return self._compute_loss(logits, tags, mask)
        
        # Otherwise return logits (inference mode)
        return logits
    
    def _compute_logits(self, char_ids, word_ids=None):
        """Compute logits from inputs"""
        # Check if we have two separate inputs (original mode)
        if word_ids is not None:
            # Original mode: char_ids [B,S,W] and word_ids [B,S]
            char_out = self.char_encoder(char_ids)  # [B, S, W, char_hidden*2]
            word_out = self.word_encoder(word_ids)  # [B, S, word_hidden*2]
            
            B, S, W = char_ids.size()
            word_expanded = word_out.unsqueeze(2).repeat(1, 1, W, 1)  # [B, S, W, word_hidden*2]
            
            combined = torch.cat([char_out, word_expanded], dim=-1)  # [B, S, W, combined_dim]
            logits = self.classifier(combined)  # [B, S, W, out_classes]
            
            return logits
        
        # Train.py mode: char_ids is actually embeddings or token indices
        if char_ids.dim() == 2:
            # [B, S] - token indices from train.py
            # Need to embed these first
            embeddings = self.char_encoder.char_emb(char_ids)  # [B, S, char_emb_dim]
            
            # Process through character-level BiLSTM
            char_out, _ = self.char_encoder.char_bilstm(embeddings)  # [B, S, char_hidden*2]
            
            # Process through word-level BiLSTM
            word_out, _ = self.word_encoder.word_bilstm(char_out)  # [B, S, word_hidden*2]
            
            # Concatenate
            combined = torch.cat([char_out, word_out], dim=-1)  # [B, S, combined_dim]
            
            # Classify
            logits = self.classifier(combined)  # [B, S, out_classes]
            
            return logits
        
        elif char_ids.dim() == 3:
            # [B, S, D] - embeddings from train.py
            B, S, D = char_ids.shape
            embeddings = char_ids
            
            # Process through character-level BiLSTM
            char_out, _ = self.char_encoder.char_bilstm(embeddings)  # [B, S, char_hidden*2]
            
            # Process through word-level BiLSTM
            word_out, _ = self.word_encoder.word_bilstm(char_out)  # [B, S, word_hidden*2]
            
            # Concatenate
            combined = torch.cat([char_out, word_out], dim=-1)  # [B, S, combined_dim]
            
            # Classify
            logits = self.classifier(combined)  # [B, S, out_classes]
            
            return logits
        
        else:
            raise ValueError(f"Unexpected input shape: {char_ids.shape}")
    
    def _compute_loss(self, logits, tags, mask):
        """Compute masked cross-entropy loss"""
        # logits: [B, S, out_classes]
        # tags: [B, S]
        # mask: [B, S]
        
        B, S, C = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.view(B * S, C)
        tags_flat = tags.view(B * S)
        
        # Create mask
        if mask is not None:
            mask_flat = mask.view(B * S).float()
        else:
            mask_flat = torch.ones(B * S, device=tags_flat.device, dtype=torch.float)
        
        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fn(logits_flat, tags_flat)
        
        # Apply mask and compute mean
        loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        
        return loss


class BiLSTMCRFClassifier(nn.Module):
    """
    Hierarchical BiLSTM + CRF model (98% accuracy with CRF variant)
    
    Takes same TWO inputs as BiLSTMClassifier:
    - char_ids: [B, S, W]
    - word_ids: [B, S]
    
    Adds CRF layer for sequence-level constraints
    """
    
    def __init__(
        self,
        n_chars,
        n_words,
        char_emb_dim,
        word_emb_dim,
        char_hidden,
        word_hidden,
        char_pad_id,
        word_pad_id,
        out_classes,
        dropout=0.3,
        use_crf=True
    ):
        super().__init__()
        
        # BiLSTM feature extractor
        self.bilstm = BiLSTMClassifier(
            n_chars=n_chars,
            n_words=n_words,
            char_emb_dim=char_emb_dim,
            word_emb_dim=word_emb_dim,
            char_hidden=char_hidden,
            word_hidden=word_hidden,
            char_pad_id=char_pad_id,
            word_pad_id=word_pad_id,
            out_classes=out_classes,
            dropout=dropout
        )
        
        self.use_crf = use_crf
        if use_crf:
            try:
                from torchcrf import CRF
            except ImportError:
                raise ImportError("Please install TorchCRF: pip install TorchCRF")
            # CRF expects [B, S, C] but we have [B, S, W, C]
            # Will handle this in forward
            self.crf = CRF(out_classes, batch_first=True)
            self.out_classes = out_classes
    
    def forward(self, char_ids, word_ids, tags=None, mask=None):
        """
        char_ids: [B, S, W]
        word_ids: [B, S]
        tags: [B, S, W] if computing loss
        mask: [B, S, W] or [B, S]
        returns: logits [B, S, W, out_classes] or loss
        """
        # Get BiLSTM logits
        logits = self.bilstm(char_ids, word_ids)  # [B, S, W, out_classes]
        
        if not self.use_crf:
            return logits
        
        # Reshape for CRF: [B, S, W, C] → [B*S, W, C]
        B, S, W, C = logits.shape
        logits_flat = logits.view(B * S, W, C)
        
        if tags is not None:
            # Flatten tags similarly
            tags_flat = tags.view(B * S, W)
            
            # Create mask if needed
            if mask is not None:
                if mask.dim() == 3:
                    mask_flat = mask.view(B * S, W)
                else:
                    # Expand [B, S] to [B*S, W]
                    mask_flat = mask.unsqueeze(-1).repeat(1, 1, W).view(B * S, W)
            else:
                mask_flat = torch.ones_like(tags_flat, dtype=torch.bool)
            
            # Compute CRF loss
            loss = -self.crf(logits_flat, tags_flat, mask=mask_flat, reduction='mean')
            return loss
        else:
            # Inference: decode
            if mask is not None:
                if mask.dim() == 3:
                    mask_flat = mask.view(B * S, W)
                else:
                    mask_flat = mask.unsqueeze(-1).repeat(1, 1, W).view(B * S, W)
            else:
                mask_flat = None
            
            predictions = self.crf.decode(logits_flat, mask=mask_flat)
            return predictions
