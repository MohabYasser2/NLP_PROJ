import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.bilstm_crf import BiLSTMCRF
from utils.vocab import CharVocab

def pad_sequences(seqs, pad_value=0):
    max_len = max(len(s) for s in seqs)
    padded = []
    mask = []

    for s in seqs:
        padded.append(s + [pad_value] * (max_len - len(s)))
        mask.append([1] * len(s) + [0] * (max_len - len(s)))

    return torch.tensor(padded), torch.tensor(mask, dtype=torch.bool)

# Example usage (you'll need to load your actual data)
# Assuming X_train, y_train are your tokenized sequences and labels

# vocab = CharVocab()
# vocab.build(X_train)
# X_encoded = [vocab.encode(seq) for seq in X_train]

# X_padded, mask = pad_sequences(X_encoded)
# y_padded, _ = pad_sequences(y_train, pad_value=-1)  # CRF uses -1 for padding

# dataset = TensorDataset(X_padded, y_padded, mask)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = BiLSTMCRF(vocab_size=len(vocab.char2id), tagset_size=15)  # 15 diacritic classes

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0

#     for X_batch, y_batch, mask_batch in train_loader:
#         optimizer.zero_grad()

#         loss = model(
#             X_batch,
#             tags=y_batch,
#             mask=mask_batch
#         )

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch}: loss = {total_loss:.4f}")