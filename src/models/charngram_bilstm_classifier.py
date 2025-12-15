"""
Character + N-gram BiLSTM Classifier (Improved)

Architecture: (Character embeddings + N-gram embeddings) → BiLSTM → Linear classifier
Same architecture as char_bilstm_classifier but with additional n-gram features.

Features:
- Character embeddings (128-dim)
- Character bigram embeddings (64-dim)

Expected Performance: 88-92% accuracy
Training Time: 2-3 hours on CPU
"""

import torch
import torch.nn as nn


class CharNgramBiLSTMClassifier(nn.Module):
    """
    Character + N-gram BiLSTM classifier.
    
    Architecture:
        Char IDs → Char Embedding (128) ─┐
                                          ├─ Concat (192) → BiLSTM (2 layers, 256) → Dropout → Linear → Softmax
        Ngram IDs → Ngram Embedding (64) ─┘
    
    The n-gram features capture local character patterns and morphology.
    
    Args:
        char_vocab_size: Character vocabulary size
        ngram_vocab_size: N-gram vocabulary size
        tagset_size: Number of diacritic classes (15)
        char_embedding_dim: Character embedding dimension (128)
        ngram_embedding_dim: N-gram embedding dimension (64)
        hidden_dim: BiLSTM hidden dimension (256)
        num_layers: Number of BiLSTM layers (2)
        dropout: Dropout rate (0.5)
    """
    
    def __init__(
        self,
        char_vocab_size,
        ngram_vocab_size,
        tagset_size,
        char_embedding_dim=128,
        ngram_embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ):
        super().__init__()
        
        # Character embedding
        self.char_embedding = nn.Embedding(
            char_vocab_size,
            char_embedding_dim,
            padding_idx=0
        )
        
        # N-gram embedding
        self.ngram_embedding = nn.Embedding(
            ngram_vocab_size,
            ngram_embedding_dim,
            padding_idx=0
        )
        
        # Fusion dimension
        fusion_dim = char_embedding_dim + ngram_embedding_dim  # 192
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            fusion_dim,
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
    
    def forward(self, char_ids, ngram_ids, tags=None, mask=None):
        """
        Forward pass with dual inputs
        
        Args:
            char_ids: Character IDs (batch, seq_len)
            ngram_ids: N-gram IDs (batch, seq_len)
            tags: Target tags (batch, seq_len) - only for training
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Training: (logits, loss)
            Inference: (logits, predictions)
        """
        # Embed characters and n-grams
        char_emb = self.char_embedding(char_ids)    # (batch, seq, 128)
        ngram_emb = self.ngram_embedding(ngram_ids)  # (batch, seq, 64)
        
        # Concatenate features (fusion)
        fused = torch.cat([char_emb, ngram_emb], dim=-1)  # (batch, seq, 192)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(fused)  # (batch, seq, 256)
        lstm_out = self.dropout(lstm_out)
        
        # Classify
        logits = self.fc(lstm_out)  # (batch, seq, tagset_size)
        
        if tags is not None:
            # Training: compute cross-entropy loss
            # Flatten for loss computation
            logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, tagset_size)
            tags_flat = tags.view(-1)  # (batch*seq)
            
            # Cross-entropy loss
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
