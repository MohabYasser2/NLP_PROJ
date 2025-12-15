import torch
import torch.nn as nn
from TorchCRF import CRF # type: ignore

class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=100,
        hidden_dim=256,
        use_contextual=False
    ):
        super().__init__()

        self.use_contextual = use_contextual

        if not use_contextual:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            lstm_input_dim = embedding_dim
        else:
            # For contextual embeddings, input is already embedded
            lstm_input_dim = embedding_dim  # Should match contextual embedding dim (768)

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, tagset_size)

        self.crf = CRF(tagset_size)

    def forward(self, x, tags=None, mask=None):
        if self.use_contextual:
            # x is already embeddings: (batch, seq, embedding_dim)
            emb = x
        else:
            # x is character indices: (batch, seq)
            emb = self.embedding(x)

        lstm_out, _ = self.lstm(emb)
        emissions = self.fc(lstm_out)

        if tags is not None:
            # Training: TorchCRF expects (seq_len, batch_size, num_tags)
            # Our model is batch_first, so we need to transpose
            emissions_transposed = emissions.transpose(0, 1)  # (batch, seq, tags) -> (seq, batch, tags)
            tags_transposed = tags.transpose(0, 1) if tags is not None else None  # (batch, seq) -> (seq, batch)
            mask_transposed = mask.transpose(0, 1) if mask is not None else None  # (batch, seq) -> (seq, batch)

            # CRF returns negative log-likelihood for each sequence in batch
            # Sum them to get total loss for the batch
            return -self.crf(emissions_transposed, tags_transposed, mask=mask_transposed).sum()
        else:
            # Inference: TorchCRF expects (seq_len, batch_size, num_tags)
            emissions_transposed = emissions.transpose(0, 1)  # (batch, seq, tags) -> (seq, batch, tags)
            mask_transposed = mask.transpose(0, 1) if mask is not None else None  # (batch, seq) -> (seq, batch)

            # CRF returns list of predictions for each sequence in batch
            predictions = self.crf.viterbi_decode(emissions_transposed, mask=mask_transposed)
            return predictions