# -*- coding: utf-8 -*-
"""
src/features/contextual_embeddings.py

Contextual Embeddings for Arabic diacritization (character-aligned).

Goal:
- Produce a contextual vector for EACH CHARACTER in your diacritization sequence.
- Works with your pipeline where training labels are per-character (tashkīl per letter).
- Handles word boundaries and expands word-level embeddings to characters.
- Uses a HuggingFace transformer (e.g., "aubmindlab/bert-base-arabertv02").

Key design choices (to fit your project):
- The transformer is fed UNDIACTRITIZED text by default (strip tashkīl), because most Arabic LMs are trained on plain Arabic.
- Alignment is done by splitting into words, encoding the full sentence, then assigning each character the embedding of its word.
  (This avoids fragile char-level offset alignment across tokenization schemes.)
- Caching supported so you don’t re-run transformer every epoch.

Dependencies:
- pip install transformers torch

Typical use:
    from src.features.contextual_embeddings import ContextualEmbedder
    embedder = ContextualEmbedder(model_name="aubmindlab/bert-base-arabertv02", device="cuda")
    vecs = embedder.embed_line_chars(clean_line)  # shape: (num_chars_no_spaces, hidden_size)

Testing:
    python -m src.features.contextual_embeddings --input data/train.txt --max_lines 2
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

# Optional import (give friendly error if missing)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModel = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# -----------------------------
# Arabic unicode helpers
# -----------------------------
# tashkīl (harakat + shadda + sukun + tanween) + dagger alif
ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL_RE = re.compile(r"\u0640")

def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics and tatweel; keep letters/spaces."""
    text = TATWEEL_RE.sub("", text)
    text = ARABIC_DIACRITICS_RE.sub("", text)
    return text

def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# Config
# -----------------------------
@dataclass
class ContextualConfig:
    model_name: str = "aubmindlab/bert-base-arabertv02"
    device: str = "cuda"  # "cpu" or "cuda"
    max_length: int = 256  # transformer max tokens
    agg: str = "last4_mean"  # "last", "last4_mean"
    strip_tashkeel: bool = True
    cache_dir: Optional[str] = None  # Set to None to disable caching (saves disk space on Kaggle)
    batch_size: int = 64  # Large batch for GPU efficiency (was 8)
    fp16: bool = False  # if cuda + supports, can speed up


# -----------------------------
# Main embedder
# -----------------------------
class ContextualEmbedder:
    def __init__(self, config: Optional[ContextualConfig] = None, **kwargs):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Missing dependencies for contextual embeddings. "
                "Install with: pip install transformers torch\n"
                f"Original import error: {_IMPORT_ERROR}"
            )

        self.config = config or ContextualConfig()
        # allow override via kwargs
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        # cache
        self.cache_dir = self.config.cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._write_meta()

    def _write_meta(self) -> None:
        """Write cache metadata so you can sanity-check which model produced embeddings."""
        meta_path = os.path.join(self.cache_dir, "meta.json")
        meta = {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "agg": self.config.agg,
            "strip_tashkeel": self.config.strip_tashkeel,
        }
        # only write if absent (don’t overwrite on upgrades)
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Public API
    # -----------------------------
    def embed_line_chars(self, line: str) -> np.ndarray:
        """
        Return contextual embeddings aligned to NON-SPACE characters in the input line.
        shape: (num_chars_without_spaces, hidden_size)
        """
        line_proc = strip_diacritics(line) if self.config.strip_tashkeel else line
        line_proc = line_proc.strip()

        if not line_proc:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        # cache hit
        cache_key = stable_hash(f"{self.config.model_name}|{self.config.agg}|{line_proc}")
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)

        # compute word embeddings then expand to chars
        char_vecs = self._embed_chars_by_word(line_proc)

        if self.cache_dir:
            np.save(cache_path, char_vecs)

        return char_vecs

    def embed_corpus_chars(self, lines: List[str]) -> List[np.ndarray]:
        """
        Batch embed many lines with TRUE GPU BATCHING for 10x speedup.
        Returns list of arrays, one per line.
        Each array is (num_chars_without_spaces, hidden_size).
        """
        import torch
        from tqdm import tqdm
        
        outputs: List[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        
        # Process in GPU-optimized batches
        for i in tqdm(range(0, len(lines), bs), desc="GPU batching", leave=False):
            chunk = lines[i:i+bs]
            
            # Strip diacritics for all lines in batch
            chunk_proc = [strip_diacritics(line) if self.config.strip_tashkeel else line 
                          for line in chunk]
            
            # Batch encode with padding
            encodings = self.tokenizer(
                chunk_proc,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Run batch through model on GPU
            with torch.no_grad():
                out = self.model(**encodings)
            
            # Extract embeddings for each line in batch
            for batch_idx, line in enumerate(chunk_proc):
                # Get token embeddings for this line
                if self.config.agg == "last":
                    token_embeds = out.last_hidden_state[batch_idx].cpu().numpy()
                else:  # last4_mean
                    all_layers = out.hidden_states[-4:]
                    stacked = torch.stack(all_layers, dim=0)
                    token_embeds = stacked.mean(dim=0)[batch_idx].cpu().numpy()
                
                # Expand to character level
                char_vecs = self._expand_tokens_to_chars(line, token_embeds, encodings, batch_idx)
                outputs.append(char_vecs)
        
        return outputs
    
    def _expand_tokens_to_chars(self, line, token_embeds, encodings, batch_idx):
        """Expand token embeddings to character level"""
        words = [w for w in line.split() if w]
        if not words:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        
        # Get word-level embeddings
        word_embs = []
        tokens_for_line = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][batch_idx])
        
        word_idx = 0
        token_idx = 1  # Skip [CLS]
        
        for word in words:
            word_token_embeds = []
            while token_idx < len(tokens_for_line) and tokens_for_line[token_idx] not in ['[SEP]', '[PAD]']:
                token_text = tokens_for_line[token_idx].replace('##', '')
                word_token_embeds.append(token_embeds[token_idx])
                token_idx += 1
                if len(''.join([t.replace('##', '') for t in tokens_for_line[1:token_idx]])) >= len(''.join(words[:word_idx+1])):
                    break
            
            if word_token_embeds:
                word_emb = np.mean(word_token_embeds, axis=0)
            else:
                word_emb = token_embeds[max(1, token_idx-1)]
            
            word_embs.append(word_emb)
            word_idx += 1
        
        # Expand each word embedding to its characters
        char_vecs = []
        for word, emb in zip(words, word_embs):
            for ch in word:
                char_vecs.append(emb)
        
        if not char_vecs:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        
        return np.array(char_vecs, dtype=np.float32)

    @property
    def hidden_size(self) -> int:
        # safest way: read from model config if exists
        return int(getattr(self.model.config, "hidden_size", 768))

    # -----------------------------
    # Internal: word -> char expansion
    # -----------------------------
    def _embed_chars_by_word(self, line_no_diac: str) -> np.ndarray:
        """
        Strategy:
        - Split the sentence into words.
        - Run transformer on whole sentence (keeps context).
        - Get token embeddings (last or last4 mean).
        - Compute one embedding per word by averaging token embeddings that belong to that word.
        - Expand each word embedding to each of its non-space characters.
        """
        # Split into words, keep exact words
        words = [w for w in line_no_diac.split() if w]
        if not words:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        # Encode full sentence
        enc = self.tokenizer(
            line_no_diac,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()  # list of (start,end) in original string

        with torch.no_grad():
            if self.config.fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        token_vecs = self._aggregate_token_vectors(out)  # (seq_len, hidden)
        token_vecs = token_vecs.detach().cpu().float().numpy()

        # Build word spans in character indices on the ORIGINAL sentence string.
        # Example: "قال أبو" -> words ["قال","أبو"]
        word_spans = self._compute_word_spans(line_no_diac)  # list[(start,end)] per word

        # Map tokens to words by overlap with word spans (using offset_mapping)
        word_embs: List[np.ndarray] = []
        for (ws, we) in word_spans:
            idxs = []
            for ti, (ts, te) in enumerate(offsets):
                # ignore special tokens with (0,0) offsets in many tokenizers
                if ts == 0 and te == 0:
                    continue
                # token overlaps word span
                if te <= ws:
                    continue
                if ts >= we:
                    continue
                idxs.append(ti)

            if not idxs:
                # fallback: zero vector (rare)
                word_embs.append(np.zeros((self.hidden_size,), dtype=np.float32))
            else:
                word_embs.append(token_vecs[idxs].mean(axis=0).astype(np.float32))

        # Expand each word embedding to its characters (excluding spaces)
        char_vecs: List[np.ndarray] = []
        for w, wv in zip(words, word_embs):
            # Each character in word gets same word vector
            for _ch in w:
                char_vecs.append(wv)

        if not char_vecs:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        return np.stack(char_vecs, axis=0)

    def _aggregate_token_vectors(self, model_output) -> "torch.Tensor":
        """
        Return token vectors of shape (seq_len, hidden).
        agg:
          - "last"        : last_hidden_state
          - "last4_mean"  : mean of last 4 hidden layers
        """
        agg = (self.config.agg or "last").lower().strip()

        if agg == "last":
            return model_output.last_hidden_state[0]  # (seq_len, hidden)

        if agg == "last4_mean":
            hs = model_output.hidden_states  # tuple layers: (emb, layer1..layerN)
            # take last 4 layers
            last4 = hs[-4:]
            stacked = torch.stack([x[0] for x in last4], dim=0)  # (4, seq_len, hidden)
            return stacked.mean(dim=0)

        # fallback
        return model_output.last_hidden_state[0]

    @staticmethod
    def _compute_word_spans(text: str) -> List[Tuple[int, int]]:
        """
        Return (start,end) spans of words in the original string.
        Words are separated by whitespace.
        """
        spans: List[Tuple[int, int]] = []
        n = len(text)
        i = 0
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            end = i
            spans.append((start, end))
        return spans


# -----------------------------
# CLI / quick test
# -----------------------------
def _read_lines(path: str, max_lines: Optional[int] = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if max_lines is not None:
        lines = lines[:max_lines]
    return lines


def _preview(line: str, vecs: np.ndarray, max_chars: int = 40) -> None:
    # Show a quick alignment check (non-space chars only)
    raw = strip_diacritics(line).replace(" ", "")
    raw = raw[:max_chars]
    show = min(len(raw), vecs.shape[0], max_chars)
    print("Text (first chars):", raw[:show])
    print("Embeddings shape:", vecs.shape)
    if show > 0:
        print("First char embedding (first 8 dims):", vecs[0][:8])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate contextual embeddings aligned to characters.")
    parser.add_argument("--input", required=True, help="Path to input text file (cleaned or raw).")
    parser.add_argument("--max_lines", type=int, default=3, help="How many lines to preview.")
    parser.add_argument("--model_name", type=str, default="aubmindlab/bert-base-arabertv02")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--agg", type=str, default="last4_mean", choices=["last", "last4_mean"])
    parser.add_argument("--no_strip", action="store_true", help="Do not strip diacritics before transformer.")
    parser.add_argument("--cache_dir", type=str, default="data/processed/contextual_cache")

    args = parser.parse_args()

    cfg = ContextualConfig(
        model_name=args.model_name,
        device=args.device,
        agg=args.agg,
        strip_tashkeel=(not args.no_strip),
        cache_dir=args.cache_dir,
    )

    embedder = ContextualEmbedder(cfg)

    lines = _read_lines(args.input, max_lines=args.max_lines)

    print("=== Contextual Embedding Preview ===")
    print("Model:", cfg.model_name)
    print("Device:", embedder.device)
    print("Agg:", cfg.agg, "| strip_tashkeel:", cfg.strip_tashkeel)
    print("Cache:", cfg.cache_dir)
    print()

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        vecs = embedder.embed_line_chars(line)
        print(f"Line {i}:")
        _preview(line, vecs)
        print("-" * 60)
