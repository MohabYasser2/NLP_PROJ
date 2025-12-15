"""
N-gram Feature Extractor for Arabic Diacritization

Extracts character n-gram features (bigrams, trigrams) to capture
local character patterns and morphological information.
"""

import torch
import torch.nn as nn
from collections import defaultdict


class NgramExtractor:
    """
    Extract character n-gram features for Arabic text.
    
    N-grams capture local character patterns like:
    - Common letter combinations (ال, من, etc.)
    - Morphological prefixes/suffixes
    - Character context windows
    """
    
    def __init__(self, n=2):
        """
        Args:
            n: N-gram size (2=bigram, 3=trigram)
        """
        self.n = n
        self.ngram2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2ngram = {0: "<PAD>", 1: "<UNK>"}
    
    def build_vocab(self, sequences):
        """
        Build n-gram vocabulary from character sequences.
        
        Args:
            sequences: List of character sequences
        """
        ngram_counts = defaultdict(int)
        
        for seq in sequences:
            ngrams = self.extract_ngrams(seq)
            for ng in ngrams:
                ngram_counts[ng] += 1
        
        # Keep n-grams that appear at least 5 times
        frequent_ngrams = [ng for ng, count in ngram_counts.items() if count >= 5]
        
        # Build vocabulary
        for ng in sorted(frequent_ngrams):
            if ng not in self.ngram2id:
                idx = len(self.ngram2id)
                self.ngram2id[ng] = idx
                self.id2ngram[idx] = ng
        
        print(f"Built {self.n}-gram vocabulary: {len(self.ngram2id)} n-grams")
    
    def extract_ngrams(self, sequence):
        """
        Extract n-grams from a character sequence.
        
        Args:
            sequence: List or string of characters
        
        Returns:
            List of n-gram strings
        """
        if len(sequence) < self.n:
            return []
        
        ngrams = []
        for i in range(len(sequence) - self.n + 1):
            ngram = ''.join(sequence[i:i+self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def encode(self, sequence):
        """
        Convert character sequence to n-gram IDs.
        For each character position, use the n-gram ending at that position.
        
        Args:
            sequence: List or string of characters
        
        Returns:
            List of n-gram IDs (same length as sequence)
        """
        if len(sequence) == 0:
            return []
        
        ngrams = self.extract_ngrams(sequence)
        
        # Pad beginning with <UNK>
        ngram_ids = [self.ngram2id.get("<UNK>")] * (self.n - 1)
        
        # Add n-gram IDs for each position
        for ng in ngrams:
            ngram_ids.append(self.ngram2id.get(ng, self.ngram2id["<UNK>"]))
        
        return ngram_ids
    
    def vocab_size(self):
        """Return vocabulary size"""
        return len(self.ngram2id)


class NgramEmbedding(nn.Module):
    """
    N-gram embedding layer.
    Embeds n-gram features for character-level modeling.
    """
    
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        """
        Args:
            vocab_size: N-gram vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding token ID
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
    
    def forward(self, ngram_ids):
        """
        Args:
            ngram_ids: (batch, seq_len) tensor of n-gram IDs
        
        Returns:
            (batch, seq_len, embedding_dim) n-gram embeddings
        """
        return self.embedding(ngram_ids)
