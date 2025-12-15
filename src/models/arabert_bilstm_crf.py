#!/usr/bin/env python3
"""
AraBERT-BiLSTM-CRF Model for Arabic Diacritization

Uses AraBERT for contextual embeddings + BiLSTM + CRF
"""

import torch
import torch.nn as nn
from TorchCRF import CRF


class AraBERTBiLSTMCRF(nn.Module):
    """
    AraBERT + BiLSTM + CRF model for Arabic diacritization

    Architecture:
    1. AraBERT encoder for contextual embeddings
    2. BiLSTM for sequence modeling
    3. CRF for structured prediction
    """
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=768,  # AraBERT hidden size
        hidden_dim=256,
        num_layers=1,
        dropout=0.3,
        freeze_bert=True
    ):
        super().__init__()

        self.freeze_bert = freeze_bert

        # AraBERT provides embeddings, so we don't need embedding layer
        # But we keep vocab_size for compatibility

        # BiLSTM encoder
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Since bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # CRF layer
        self.crf = CRF(tagset_size)

        # Optional linear layer to adjust dimensions if needed
        self.proj = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, tags=None, mask=None):
        """
        Forward pass

        Args:
            x: AraBERT embeddings (batch, seq_len, 768)
            tags: Target labels for training (batch, seq_len)
            mask: Attention mask (batch, seq_len)

        Returns:
            Loss for training, predictions for inference
        """
        # x is already AraBERT embeddings: (batch, seq_len, embedding_dim)

        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden_dim)

        # Project to tag space
        emissions = self.proj(lstm_out)  # (batch, seq_len, tagset_size)

        if tags is not None:
            # Training: CRF loss
            # TorchCRF expects (seq_len, batch_size, num_tags)
            emissions_transposed = emissions.transpose(0, 1)  # (seq, batch, tags)
            tags_transposed = tags.transpose(0, 1) if tags is not None else None
            mask_transposed = mask.transpose(0, 1) if mask is not None else None

            # CRF returns negative log-likelihood
            return -self.crf(emissions_transposed, tags_transposed, mask=mask_transposed).sum()
        else:
            # Inference: CRF decoding
            emissions_transposed = emissions.transpose(0, 1)
            mask_transposed = mask.transpose(0, 1) if mask is not None else None

            # CRF returns list of predictions
            predictions = self.crf.viterbi_decode(emissions_transposed, mask=mask_transposed)
            return predictions

    def freeze_arabert(self):
        """Freeze AraBERT parameters if needed"""
        # This would be called externally on the AraBERT model
        pass

    def unfreeze_arabert(self):
        """Unfreeze AraBERT parameters for fine-tuning"""
        pass