import torch

def pad_sequences(seqs, pad_value=0):
    """
    Pad sequences to the same length.

    Args:
        seqs: List of sequences (lists of ints)
        pad_value: Value to pad with

    Returns:
        padded: Tensor of shape (batch_size, max_len)
        mask: Boolean tensor of shape (batch_size, max_len), True for real tokens
    """
    max_len = max(len(s) for s in seqs)
    padded = []
    mask = []

    for s in seqs:
        padded.append(s + [pad_value] * (max_len - len(s)))
        mask.append([1] * len(s) + [0] * (max_len - len(s)))

    return torch.tensor(padded), torch.tensor(mask, dtype=torch.bool)