# -*- coding: utf-8 -*-
"""
Arabic Diacritization Model with Bi-LSTM
Improved version with optimized hyperparameters
"""

import os
# Fix for OpenMP duplicate library error (common with Anaconda)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =====================
# ARGPARSE FOR PATHS
# =====================
import argparse
parser = argparse.ArgumentParser(description="Arabic Diacritization Model with Bi-LSTM")
parser.add_argument('--train_path', type=str, default="c:/Users/mohab/Desktop/NLP/WorkingModels/train.txt", help='Path to training file')
parser.add_argument('--val_path', type=str, default="c:/Users/mohab/Desktop/NLP/WorkingModels/val.txt", help='Path to validation file')
parser.add_argument('--model_save_path', type=str, default="c:/Users/mohab/Desktop/NLP/WorkingModels/best_model_Bi-LSTM.pth", help='Path to save best model')
args = parser.parse_args()

# Paths from argparse or defaults
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
MODEL_SAVE_PATH = args.model_save_path

import io
import random
import sys
from collections import Counter
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE TO YOUR LOCAL PATHS
TRAIN_PATH = "C:\\Users\\mohab\\Desktop\\NLP\\WorkingModels\\train.txt"  # Path to your training file
VAL_PATH = "C:\\Users\\mohab\\Desktop\\NLP\\WorkingModels\\val.txt"      # Path to your validation file
MODEL_SAVE_PATH = "C:\\Users\\mohab\\Desktop\\NLP\\WorkingModels\\best_model_Bi-LSTM.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def clean_reference_markers(text):
    """Remove reference markers like (20/8), (1325), etc."""
    # Remove all content inside (), [], {}
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    # Remove colons, semicolons, and normalize Arabic punctuation
    text = re.sub(r'[:؛;]', '', text)
    # Remove any non-Arabic, non-diacritic, non-space, non-Arabic punctuation chars
    # Arabic unicode: \u0600-\u06FF, diacritics \u0610-\u061A\u064B-\u0652, Arabic punctuation \u060C\u061B\u061F
    text = re.sub(r'[^\u0600-\u06FF\u0610-\u061A\u064B-\u0652\u060C\u061B\u061F\s]', '', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading training data...")
with open(TRAIN_PATH, "r", encoding="utf8") as f:
    lines = f.read().splitlines()

cleaned_lines = []
for line in lines:
    cleaned = clean_reference_markers(line)
    if cleaned.strip():
        cleaned_lines.append(cleaned)

print(f"Loaded {len(cleaned_lines)} lines from training file")

# Arabic sentence delimiters
SPLIT_REGEX = r"(?<=[\.])"

def split_into_sentences(text):
    """Split text by punctuation but keep the delimiter"""
    parts = re.split(SPLIT_REGEX, text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

sentences = []
for line in cleaned_lines:
    sents = split_into_sentences(line)
    sentences.extend(sents)

print(f"After punctuation splitting: {len(sentences)} sentences")

# Split long sentences by word limit
def split_by_word_limit(sentence, max_words=80):
    """Split sentences longer than max_words"""
    words = sentence.split()
    parts = []
    current = []
    
    for w in words:
        if len(current) < max_words:
            current.append(w)
        else:
            parts.append(" ".join(current))
            current = [w]
    if current:
        parts.append(" ".join(current))
    
    return parts

processed = []
for s in sentences:
    wc = len(s.split())
    if wc > 80:
        processed.extend(split_by_word_limit(s, 80))
    else:
        processed.append(s)

print(f"Total processed sentences: {len(processed)}")

# ============================================================================
# DIACRITICS REMOVAL
# ============================================================================

DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u0652]')

def remove_diacritics(text):
    """Remove all diacritics from Arabic text"""
    return DIACRITICS.sub('', text)

X = []  # undiacritized sentences
Y = []  # diacritized sentences

for s in processed:
    Y.append(s)
    X.append(remove_diacritics(s))

print(f"Created {len(X)} training pairs (X, Y)")

# ============================================================================
# TOKENIZATION (requires 'regex' package)
# ============================================================================

try:
    import regex as re_unicode
except ImportError:
    print("Installing regex package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "regex"])
    import regex as re_unicode


# Load Arabic letters and diacritics from pickle files in utils
import pickle
ARABIC_LETTERS = pickle.load(open('c:/Users/mohab/Desktop/NLP/WorkingModels/utils/arabic_letters.pickle', 'rb'))
DIACRITICS_SET = pickle.load(open('c:/Users/mohab/Desktop/NLP/WorkingModels/utils/diacritics.pickle', 'rb'))

# Build regex patterns from loaded sets
AR_LETTER = '[' + ''.join(ARABIC_LETTERS) + ']'
DIAC = '[' + ''.join(DIACRITICS_SET) + ']'
AR_WORD = rf'(?:{AR_LETTER}{DIAC}*)+'
WORD_TOKENIZER = re_unicode.compile(rf"{AR_WORD}|[0-9]+|[^\s]")

def tokenize_words(sentence):
    """Tokenize sentence into words"""
    return WORD_TOKENIZER.findall(sentence)

X_words = []
Y_words = []

for x, y in zip(X, Y):
    X_words.append(tokenize_words(x))
    Y_words.append(tokenize_words(y))

print("Tokenization complete")

# ============================================================================
# CHARACTER-LEVEL PROCESSING
# ============================================================================

AR_DIACRITICS = r"[\u0610-\u061A\u064B-\u0652]"

def split_chars_with_diacritics(word):
    """Split word into individual characters (including diacritics)"""
    return list(word)

def split_chars_no_diac(word):
    """Split undiacritized word into characters"""
    return list(word)

X_chars = []
Y_chars = []

for x_sent, y_sent in zip(X_words, Y_words):
    x_sent_chars = []
    y_sent_chars = []
    
    for xw, yw in zip(x_sent, y_sent):
        x_sent_chars.append(split_chars_no_diac(xw))
        y_sent_chars.append(split_chars_with_diacritics(yw))
    
    X_chars.append(x_sent_chars)
    Y_chars.append(y_sent_chars)

# ============================================================================
# LABEL EXTRACTION
# ============================================================================


# Load DIACRITIC_LABELS from pickle file in utils
import pickle
DIACRITIC_LABELS = pickle.load(open('c:/Users/mohab/Desktop/NLP/WorkingModels/utils/diacritic2id.pickle', 'rb'))

id2diac = {idx: diac for diac, idx in DIACRITIC_LABELS.items()}

def extract_labels_from_ychars(y_chars):
    """Extract base letters and diacritic labels from character list"""
    base_letters = []
    labels = []
    
    current_letter = None
    current_diacritics = ""
    
    for ch in y_chars:
        if re_unicode.fullmatch(AR_DIACRITICS, ch):
            if current_letter is not None:
                current_diacritics += ch
        else:
            if current_letter is not None:
                base_letters.append(current_letter)
                labels.append(current_diacritics)
            current_letter = ch
            current_diacritics = ""
    
    if current_letter is not None:
        base_letters.append(current_letter)
        labels.append(current_diacritics)
    
    return base_letters, labels

def convert_labels_to_ids(label_list):
    """Convert diacritic strings to label IDs"""
    return [DIACRITIC_LABELS.get(lbl, 0) for lbl in label_list]

X_final = []
Y_final = []

for x_sent_chars, y_sent_chars in zip(X_chars, Y_chars):
    sent_x = []
    sent_y = []
    
    for x_word_chars, y_word_chars in zip(x_sent_chars, y_sent_chars):
        y_base_letters, y_labels = extract_labels_from_ychars(y_word_chars)
        y_label_ids = convert_labels_to_ids(y_labels)
        
        assert len(x_word_chars) == len(y_base_letters), "Alignment error!"
        
        sent_x.append(x_word_chars)
        sent_y.append(y_label_ids)
    
    X_final.append(sent_x)
    Y_final.append(sent_y)

print("Label extraction complete")

# ============================================================================
# VOCABULARY BUILDING
# ============================================================================

# Character vocabulary
char_counter = Counter()
for sent in X_final:
    for word in sent:
        for ch in word:
            char_counter[ch] += 1

CHAR_PAD = "<PAD>"
CHAR_UNK = "<UNK>"
chars = [CHAR_PAD, CHAR_UNK] + sorted(char_counter.keys())

char2id = {ch: idx for idx, ch in enumerate(chars)}
id2char = {idx: ch for ch, idx in char2id.items()}

print(f"Character vocabulary size: {len(chars)}")

# Word vocabulary
word_counter = Counter()
for sent in X_words:
    for word in sent:
        word_counter[word] += 1

WORD_PAD = "<W_PAD>"
WORD_UNK = "<W_UNK>"
words = [WORD_PAD, WORD_UNK] + sorted(word_counter.keys())

word2id = {w: i for i, w in enumerate(words)}
id2word = {i: w for w, i in word2id.items()}

print(f"Word vocabulary size: {len(words)}")

# ============================================================================
# PADDING AND TENSOR CREATION
# ============================================================================

max_sentence_len = max(len(sent) for sent in X_final)
max_word_len = max(len(word) for sent in X_final for word in sent)

print(f"Max sentence length: {max_sentence_len} words")
print(f"Max word length: {max_word_len} characters")

def pad_word_chars(word_chars):
    """Pad character sequence to max_word_len"""
    padded = word_chars[:max_word_len]
    padded = [char2id.get(ch, char2id[CHAR_UNK]) for ch in padded]
    padded += [char2id[CHAR_PAD]] * (max_word_len - len(padded))
    return padded

def pad_word_labels(labels):
    """Pad label sequence to max_word_len"""
    padded = labels[:max_word_len]
    padded += [0] * (max_word_len - len(padded))
    return padded

def pad_sentence_words(sentence_words):
    """Pad sentence to max_sentence_len"""
    padded = sentence_words[:max_sentence_len]
    padded += [[char2id[CHAR_PAD]] * max_word_len] * (max_sentence_len - len(padded))
    return padded

def pad_sentence_labels(sentence_labels):
    """Pad sentence labels to max_sentence_len"""
    padded = sentence_labels[:max_sentence_len]
    padded += [[0] * max_word_len] * (max_sentence_len - len(padded))
    return padded

# Create tensors
X_tensor = []
Y_tensor = []

for sent_x, sent_y in zip(X_final, Y_final):
    sent_x_padded = [pad_word_chars(word) for word in sent_x]
    sent_y_padded = [pad_word_labels(labels) for labels in sent_y]
    
    sent_x_padded = pad_sentence_words(sent_x_padded)
    sent_y_padded = pad_sentence_labels(sent_y_padded)
    
    X_tensor.append(sent_x_padded)
    Y_tensor.append(sent_y_padded)

X_tensor = torch.tensor(X_tensor, dtype=torch.long)
Y_tensor = torch.tensor(Y_tensor, dtype=torch.long)

print(f"X_tensor shape: {X_tensor.shape}")
print(f"Y_tensor shape: {Y_tensor.shape}")

def build_word_tensor(X_words, word2id, max_sentence_len, pad_token="<W_PAD>"):
    """Build word-level tensor"""
    pad_id = word2id.get(pad_token)
    word_tensor = []
    
    for sent in X_words:
        ids = [word2id.get(w, word2id["<W_UNK>"]) for w in sent]
        ids = ids[:max_sentence_len]
        ids += [pad_id] * (max_sentence_len - len(ids))
        word_tensor.append(ids)
    
    return torch.tensor(word_tensor, dtype=torch.long)

word_tensor = build_word_tensor(X_words, word2id, max_sentence_len)
print(f"word_tensor shape: {word_tensor.shape}")

# ============================================================================
# DATASET CLASS
# ============================================================================

class DiacriticsDataset(Dataset):
    def __init__(self, X_chars, X_words, Y_labels):
        self.X_chars = X_chars
        self.X_words = X_words
        self.Y_labels = Y_labels
    
    def __len__(self):
        return len(self.X_chars)
    
    def __getitem__(self, idx):
        return (
            self.X_chars[idx],
            self.X_words[idx],
            self.Y_labels[idx]
        )

dataset = DiacriticsDataset(X_tensor, word_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Created DataLoader with {len(dataset)} samples")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class WordLevelBiLSTM(nn.Module):
    def __init__(self, n_words, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.5):
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
        emb = self.word_emb(word_ids)
        out, _ = self.word_bilstm(emb)
        return out

class CharLevelBiLSTM(nn.Module):
    def __init__(self, n_chars, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.5):
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
        B, S, W = char_ids.size()
        flat = char_ids.view(B * S, W)
        emb = self.char_emb(flat)
        out, _ = self.char_bilstm(emb)
        out = out.view(B, S, W, -1)
        return out

class DiacriticsModel(nn.Module):
    def __init__(
        self,
        n_chars, char_emb_dim, char_hidden, char_layers, char_pad_id,
        n_words, word_emb_dim, word_hidden, word_layers, word_pad_id,
        out_classes, dropout=0.5
    ):
        super().__init__()
        
        self.char_encoder = CharLevelBiLSTM(
            n_chars=n_chars,
            emb_dim=char_emb_dim,
            hidden_dim=char_hidden,
            num_layers=char_layers,
            pad_id=char_pad_id,
            dropout=dropout
        )
        
        self.word_encoder = WordLevelBiLSTM(
            n_words=n_words,
            emb_dim=word_emb_dim,
            hidden_dim=word_hidden,
            num_layers=word_layers,
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
    
    def forward(self, char_ids, word_ids):
        char_out = self.char_encoder(char_ids)
        word_out = self.word_encoder(word_ids)
        
        B, S, W = char_ids.size()
        word_expanded = word_out.unsqueeze(2).repeat(1, 1, W, 1)
        
        combined = torch.cat([char_out, word_expanded], dim=-1)
        logits = self.classifier(combined)
        
        return logits

# Initialize model with improved hyperparameters
model = DiacriticsModel(
    n_chars=len(chars),
    char_emb_dim=128,
    char_hidden=256,
    char_layers=2,
    char_pad_id=char2id["<PAD>"],
    
    n_words=len(words),
    word_emb_dim=256,
    word_hidden=256,
    word_layers=2,
    word_pad_id=word2id["<W_PAD>"],
    
    out_classes=len(DIACRITIC_LABELS),
    dropout=0.5
).to(device)

print("Model initialized")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

def compute_accuracy(logits, labels, char_ids, pad_char_id):
    """Compute character-level accuracy"""
    preds = logits.argmax(dim=-1)
    mask = (char_ids != pad_char_id)
    
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    return correct / total

def train_step(model, char_ids, word_ids, labels, pad_char_id, optimizer):
    """Single training step"""
    model.train()
    
    char_ids = char_ids.to(device)
    word_ids = word_ids.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    
    logits = model(char_ids, word_ids)
    B, S, W, C = logits.shape
    
    logits_flat = logits.view(B * S * W, C)
    labels_flat = labels.view(B * S * W)
    mask = (char_ids.view(B * S * W) != pad_char_id).float()
    
    loss_per_char = criterion(logits_flat, labels_flat)
    loss = (loss_per_char * mask).sum() / mask.sum().clamp(min=1.0)
    
    loss.backward()
    optimizer.step()
    
    acc = compute_accuracy(logits, labels, char_ids, pad_char_id)
    
    return loss.item(), acc

# ============================================================================
# VALIDATION DATA LOADING
# ============================================================================

print("\nLoading validation data...")
with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_lines = [line.strip() for line in f if line.strip()]

DIAC_REGEX = re.compile(r"[\u0610-\u061A\u064B-\u0652]")

def strip_diacritics(text):
    return DIAC_REGEX.sub("", text)

X_val_sentences = [strip_diacritics(line) for line in val_lines]
Y_val_sentences = val_lines

def split_words(sentence):
    return sentence.split()

X_val_words = [split_words(s) for s in X_val_sentences]
Y_val_words = [split_words(s) for s in Y_val_sentences]

X_val_chars = [[list(w) for w in sent] for sent in X_val_words]
Y_val_chars = [[list(w) for w in sent] for sent in Y_val_words]

X_val_final = []
Y_val_final = []

for x_sent, y_sent in zip(X_val_chars, Y_val_chars):
    sent_x = []
    sent_y = []
    
    for x_word, y_word in zip(x_sent, y_sent):
        base_letters, labels = extract_labels_from_ychars(y_word)
        label_ids = convert_labels_to_ids(labels)
        
        assert len(x_word) == len(base_letters), "Mismatch in val!"
        
        sent_x.append(x_word)
        sent_y.append(label_ids)
    
    X_val_final.append(sent_x)
    Y_val_final.append(sent_y)

def build_char_tensor(X_final, char2id, max_sentence_len, max_word_len):
    char_tensor = []
    for sent_x in X_final:
        sent_x_padded = [pad_word_chars(word) for word in sent_x]
        sent_x_padded = pad_sentence_words(sent_x_padded)
        char_tensor.append(sent_x_padded)
    return torch.tensor(char_tensor, dtype=torch.long)

def build_label_tensor(Y_final, max_sentence_len, max_word_len):
    label_tensor = []
    for sent_y in Y_final:
        sent_y_padded = [pad_word_labels(labels) for labels in sent_y]
        sent_y_padded = pad_sentence_labels(sent_y_padded)
        label_tensor.append(sent_y_padded)
    return torch.tensor(label_tensor, dtype=torch.long)

X_val_tensor = build_char_tensor(X_val_final, char2id, max_sentence_len, max_word_len)
Y_val_tensor = build_label_tensor(Y_val_final, max_sentence_len, max_word_len)
word_val_tensor = build_word_tensor(X_val_words, word2id, max_sentence_len)

def evaluate(model, data_loader, pad_char_id):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_chars = 0
    total_incorrect_diacritics = 0
    
    with torch.no_grad():
        for batch_idx, (chars, words, labels) in enumerate(data_loader):
            chars = chars.to(device)
            words = words.to(device)
            labels = labels.to(device)
            
            logits = model(chars, words)
            B, S, W, C = logits.shape
            
            logits_flat = logits.view(B * S * W, C)
            labels_flat = labels.view(B * S * W)
            mask = (chars.view(B * S * W) != pad_char_id).float()
            
            loss_per_char = criterion(logits_flat, labels_flat)
            loss = (loss_per_char * mask).sum() / mask.sum().clamp(min=1.0)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            acc_mask = (chars != pad_char_id)
            correct = ((preds == labels) & acc_mask).sum().item()
            total = acc_mask.sum().item()
            total_acc += correct
            total_chars += total
            
            incorrect_diacritics = ((preds != labels) & acc_mask).sum().item()
            total_incorrect_diacritics += incorrect_diacritics
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / total_chars if total_chars > 0 else 0.0
    der = total_incorrect_diacritics / total_chars if total_chars > 0 else 0.0
    
    return avg_loss, avg_acc, der

val_dataset = DiacriticsDataset(X_val_tensor, word_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Validation set: {len(val_dataset)} samples")

# ============================================================================
# TRAINING LOOP
# ============================================================================

best_val_loss = float('inf')
best_val_der = float('inf')
patience = 5
patience_counter = 0

print("\n" + "=" * 60)
print("TRAINING WITH IMPROVED HYPERPARAMETERS")
print("=" * 60)

for epoch in range(1, 21):
    total_loss = 0
    total_acc = 0
    
    for batch_idx, (chars, words, labels) in enumerate(loader):
        loss, acc = train_step(
            model=model,
            char_ids=chars,
            word_ids=words,
            labels=labels,
            pad_char_id=char2id["<PAD>"],
            optimizer=optimizer
        )
        
        total_loss += loss
        total_acc += acc
        
        if batch_idx % 50 == 0:
            print(f"   batch {batch_idx}: loss={loss:.4f}, acc={acc:.4f}")
    
    epoch_loss = total_loss / len(loader)
    epoch_acc = total_acc / len(loader)
    
    val_loss, val_acc, val_der = evaluate(model, val_loader, pad_char_id=char2id["<PAD>"])
    
    print(f"\nEpoch {epoch}:")
    print(f"  Train - loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}")
    print(f"  Val   - loss={val_loss:.4f}, accuracy={val_acc:.4f}, DER={val_der:.4f}")
    print("-" * 60)
    
    if val_der < best_val_der:
        best_val_der = val_der
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved! (DER: {val_der:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation DER: {best_val_der:.4f}")
            break

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"Best validation DER: {best_val_der:.4f}")
print("=" * 60)

# Load best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print("Loaded best model weights")

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def preprocess_sentence(sentence, char2id, word2id, max_sentence_len, max_word_len):
    """Preprocess a single sentence for inference"""
    clean = strip_diacritics(sentence)
    words = clean.split()
    
    words = words[:max_sentence_len]
    while len(words) < max_sentence_len:
        words.append("<W_PAD>")
    
    word_ids = [word2id.get(w, word2id["<W_UNK>"]) for w in words]
    word_ids = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0)
    
    char_id_matrix = []
    raw_char_matrix = []
    
    for w in words:
        chars = list(w)
        raw_char_matrix.append(chars if w != "<W_PAD>" else ["<W_PAD>"])
        
        ch_ids = [char2id.get(c, char2id["<UNK>"]) for c in chars]
        ch_ids = ch_ids[:max_word_len]
        ch_ids += [char2id["<PAD>"]] * (max_word_len - len(ch_ids))
        
        char_id_matrix.append(ch_ids)
    
    char_ids = torch.tensor(char_id_matrix, dtype=torch.long).unsqueeze(0)
    
    return char_ids, word_ids, raw_char_matrix

def reconstruct(preds, raw_chars, id2diac):
    """Reconstruct diacritized text from predictions"""
    out_words = []
    
    for word_pred, word_chars in zip(preds, raw_chars):
        if len(word_chars) == 1 and word_chars[0] == "<W_PAD>":
            break
        
        word_out = ""
        for ch, diac_id in zip(word_chars, word_pred):
            if ch == "<W_PAD>":
                continue
            
            diac = id2diac[int(diac_id)]
            word_out += ch + diac
        
        out_words.append(word_out)
    
    return " ".join(out_words)

def diacritize(sentence):
    """Diacritize a sentence"""
    model.eval()
    
    char_ids, word_ids, raw_char_matrix = preprocess_sentence(
        sentence,
        char2id,
        word2id,
        max_sentence_len,
        max_word_len
    )
    
    char_ids = char_ids.to(device)
    word_ids = word_ids.to(device)
    
    with torch.no_grad():
        logits = model(char_ids, word_ids)
        preds = logits.argmax(dim=-1)[0]
    
    return reconstruct(preds, raw_char_matrix, id2diac)

# ============================================================================
# TEST THE MODEL
# ============================================================================

test_sentence = "في صباح يوم الاثنين خرج الطالب من منزله متوجها الى الجامعة وعندما وصل الى القاعة وجد ان المحاضرة قد بدأت فجلس في اخر الصف بهدوء واستمع الى شرح الاستاذ الذي كان يتحدث عن موضوع مهم يتعلق بتاريخ العلوم وتطورها عبر العصور"

print("\n" + "=" * 60)
print("TEST DIACRITIZATION")
print("=" * 60)
print("\nInput:")
print(test_sentence)
print("\nOutput:")
print(diacritize(test_sentence))
print("\n" + "=" * 60)
