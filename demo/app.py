import sys
import os
import torch
import pickle
import unicodedata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.arabert_char_bilstm_crf import AraBERTCharBiLSTMCRF
from src.models.bilstm_crf import BiLSTMCRF
from src.models.charngram_bilstm_classifier import CharNgramBiLSTMClassifier
from src.models.char_bilstm_classifier import CharBiLSTMClassifier
from src.features.contextual_embeddings import ContextualEmbedder
from src.preprocessing.tokenize import tokenize_line, is_arabic_base_letter, is_arabic_digit, is_arabic_diacritic, DIACRITIC_CHARS
from utils.vocab import CharVocab
from src.config import get_model_config

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and resources
model_cache = {}  # Cache loaded models
vocab = None
diacritic2id = None
id2diacritic = None
embedder = None
device = None
current_model_name = None

class DiacritizeRequest(BaseModel):
    text: str
    model_name: str = "best_arabert_char_bilstm_crf.pth"

def detect_model_type(model_name: str) -> str:
    """Detect model type from filename"""
    name_lower = model_name.lower()
    if "arabert_char" in name_lower or "arabert-char" in name_lower:
        return "arabert_char_bilstm_crf"
    elif "charngram" in name_lower or "char_ngram" in name_lower:
        return "charngram_bilstm_classifier"
    elif "char_bilstm_classifier" in name_lower or "char-bilstm-classifier" in name_lower:
        return "char_bilstm_classifier"
    elif "bilstm_crf" in name_lower:
        return "bilstm_crf"
    else:
        return "unknown"

def get_available_models() -> List[Dict[str, str]]:
    """Get list of available models from models folder"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")
    
    models = []
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(".pth"):
                model_type = detect_model_type(filename)
                models.append({
                    "filename": filename,
                    "type": model_type,
                    "display_name": filename.replace(".pth", "")
                })
    
    # Sort by type, then by name
    models.sort(key=lambda x: (x["type"], x["filename"]))
    return models

def load_model_architecture(model_type: str, config: dict, vocab_size: int):
    """Load appropriate model architecture based on type"""
    if model_type == "arabert_char_bilstm_crf":
        return AraBERTCharBiLSTMCRF(
            char_vocab_size=vocab_size,
            tagset_size=config.get('tagset_size', 15),
            arabert_dim=config.get('arabert_dim', 768),
            char_embedding_dim=config.get('char_embedding_dim', 100),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == "bilstm_crf":
        return BiLSTMCRF(
            vocab_size=vocab_size,
            tagset_size=config.get('tagset_size', 15),
            embedding_dim=config.get('embedding_dim', 100),
            hidden_dim=config.get('hidden_dim', 256),
            use_contextual=config.get('use_contextual', False)
        )
    elif model_type == "charngram_bilstm_classifier":
        return CharNgramBiLSTMClassifier(
            char_vocab_size=vocab_size,
            ngram_vocab_size=config.get('ngram_vocab_size', 5000),
            tagset_size=config.get('tagset_size', 15),
            char_embedding_dim=config.get('char_embedding_dim', 128),
            ngram_embedding_dim=config.get('ngram_embedding_dim', 64),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5)
        )
    elif model_type == "char_bilstm_classifier":
        return CharBiLSTMClassifier(
            vocab_size=vocab_size,
            tagset_size=config.get('tagset_size', 15),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_resources(model_name: str = "best_arabert_char_bilstm_crf.pth"):
    global model_cache, vocab, diacritic2id, id2diacritic, embedder, device, current_model_name
    
    # Return cached model if already loaded
    if model_name in model_cache:
        print(f"Using cached model: {model_name}")
        current_model_name = model_name
        return model_cache[model_name]
    
    print(f"Loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load diacritic mapping (only once)
    if diacritic2id is None:
        with open(os.path.join(BASE_DIR, "utils/diacritic2id.pickle"), "rb") as f:
            diacritic2id = pickle.load(f)
        id2diacritic = {v: k for k, v in diacritic2id.items()}

    # Load checkpoint
    model_path = os.path.join(BASE_DIR, "models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Rebuild vocab
    vocab = CharVocab()
    vocab.char2id = checkpoint['vocab']
    vocab.id2char = {v: k for k, v in vocab.char2id.items()}

    # Load config
    config = checkpoint['config']
    
    # Detect model type
    model_type = detect_model_type(model_name)
    print(f"Detected model type: {model_type}")
    
    # Initialize embedder for contextual models
    if model_type in ["arabert_char_bilstm_crf", "bilstm_crf"]:
        if config.get('use_contextual', False) or model_type == "arabert_char_bilstm_crf":
            if embedder is None:
                print("Initializing AraBERT embedder...")
                embedder = ContextualEmbedder(
                    model_name="aubmindlab/bert-base-arabertv02",
                    device=device.type,
                    cache_dir=None
                )
    
    # Load model architecture
    model = load_model_architecture(model_type, config, len(vocab.char2id))
    
    # Load state dict with parameter name mapping for CRF models
    if model_type in ["arabert_char_bilstm_crf", "bilstm_crf"]:
        state_dict = checkpoint['model_state_dict']
        # Map old CRF parameter names to new ones if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'crf.trans_matrix' in k:
                new_state_dict[k.replace('trans_matrix', 'transitions')] = v
            elif 'crf.start_trans' in k:
                new_state_dict[k.replace('start_trans', 'start_transitions')] = v
            elif 'crf.end_trans' in k:
                new_state_dict[k.replace('end_trans', 'end_transitions')] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Cache the model
    model_cache[model_name] = model
    current_model_name = model_name
    
    print(f"Model {model_name} loaded successfully!")
    return model

@app.on_event("startup")
async def startup_event():
    # Load default model on startup
    load_resources("best_arabert_char_bilstm_crf.pth")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/models")
async def list_models():
    """Get list of available models"""
    models = get_available_models()
    return {"models": models}

def reconstruct_text(line: str, predictions: list) -> str:
    """
    Reconstruct text with predicted diacritics, preserving spaces and structure.
    Mirrors logic in src/preprocessing/tokenize.py
    """
    # Normalize Unicode to canonical form (same as tokenize_line)
    line = unicodedata.normalize("NFKC", line)
    line = unicodedata.normalize("NFC", line)

    out = []
    pred_idx = 0
    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        if ch.isspace():
            out.append(ch)
            i += 1
            continue

        if is_arabic_digit(ch):
            out.append(ch)
            # Digits get empty diacritic prediction usually, but we consume it
            if pred_idx < len(predictions):
                out.append(predictions[pred_idx])
                pred_idx += 1
            i += 1
            continue

        if is_arabic_base_letter(ch):
            out.append(ch)
            
            # Skip existing diacritics in input
            i += 1
            while i < n and is_arabic_diacritic(line[i]):
                i += 1
            
            # Append predicted diacritic
            if pred_idx < len(predictions):
                out.append(predictions[pred_idx])
                pred_idx += 1
            continue

        # Anything else (punctuation, etc.) - tokenize_line skips these!
        # But for reconstruction, we might want to keep them if they weren't tokenized?
        # However, the model predictions align with tokenize_line output.
        # If tokenize_line skipped it, we have no prediction for it.
        # If we keep it, we don't increment pred_idx.
        # But tokenize_line logic is: "Anything else -> i += 1". It drops it.
        # So we should probably drop it too to stay consistent, OR keep it but know there's no prediction.
        # Let's keep it for better UX, but NOT consume a prediction.
        out.append(ch)
        i += 1

    return "".join(out)

@app.post("/diacritize")
async def diacritize(request: DiacritizeRequest):
    global current_model_name
    
    text = request.text
    model_name = request.model_name
    
    if not text:
        return {"diacritized_text": ""}

    try:
        # Load model if different from current
        if model_name != current_model_name:
            model = load_resources(model_name)
        else:
            model = model_cache.get(current_model_name)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # 1. Tokenize (to get base chars for model input)
        base_chars, _ = tokenize_line(text)
        
        if not base_chars:
             return {"diacritized_text": text}

        # Detect model type for appropriate inference
        model_type = detect_model_type(model_name)
        
        # 2. Prepare inputs based on model type
        char_ids = vocab.encode(base_chars)
        char_ids_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
        mask = torch.ones((1, len(base_chars)), dtype=torch.bool).to(device)
        
        # 3. Inference based on model type
        with torch.no_grad():
            if model_type == "arabert_char_bilstm_crf":
                # AraBERT embeddings - use base_chars joined to ensure alignment
                stripped_text = ''.join(base_chars)
                emb = embedder.embed_line_chars(stripped_text)
                import numpy as np
                emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                emb_tensor = torch.from_numpy(emb_array).unsqueeze(0).float().to(device)
                
                # Ensure sequence lengths match
                if emb_tensor.size(1) != char_ids_tensor.size(1):
                    min_len = min(emb_tensor.size(1), char_ids_tensor.size(1))
                    emb_tensor = emb_tensor[:, :min_len, :]
                    char_ids_tensor = char_ids_tensor[:, :min_len]
                    mask = mask[:, :min_len]
                
                # Forward pass (returns list of lists)
                prediction_ids = model(emb_tensor, char_ids_tensor, mask=mask)
                
            elif model_type == "bilstm_crf":
                # Check if model uses contextual embeddings
                if hasattr(model, 'use_contextual') and model.use_contextual:
                    stripped_text = ''.join(base_chars)
                    emb = embedder.embed_line_chars(stripped_text)
                    import numpy as np
                    emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                    emb_tensor = torch.from_numpy(emb_array).unsqueeze(0).float().to(device)
                    
                    # Ensure sequence lengths match
                    if emb_tensor.size(1) != char_ids_tensor.size(1):
                        min_len = min(emb_tensor.size(1), char_ids_tensor.size(1))
                        emb_tensor = emb_tensor[:, :min_len, :]
                        mask = mask[:, :min_len]
                    
                    prediction_ids = model(emb_tensor, mask=mask)
                else:
                    prediction_ids = model(char_ids_tensor, mask=mask)
                    
            elif model_type == "charngram_bilstm_classifier":
                # CharNgram model needs both char_ids and ngram_ids
                # For demo, we'll use zeros for ngrams (model will still work)
                ngram_ids = torch.zeros_like(char_ids_tensor)
                result = model(char_ids_tensor, ngram_ids)
                if isinstance(result, tuple):
                    logits, predictions = result
                    prediction_ids = predictions[0].tolist()
                else:
                    # Fallback if it returns only logits
                    logits = result
                    prediction_ids = torch.argmax(logits, dim=-1)[0].tolist()
                
            elif model_type == "char_bilstm_classifier":
                # Character-only model returns (logits, predictions) tuple
                result = model(char_ids_tensor)
                if isinstance(result, tuple):
                    logits, predictions = result
                    prediction_ids = predictions[0].tolist()
                else:
                    # Fallback if it returns only logits
                    logits = result
                    prediction_ids = torch.argmax(logits, dim=-1)[0].tolist()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # 4. Decode predictions
        if isinstance(prediction_ids, list) and len(prediction_ids) > 0 and isinstance(prediction_ids[0], list):
            # CRF models return list of lists
            predicted_diacritics = [id2diacritic[pid] for pid in prediction_ids[0]]
        else:
            # Classifier models return flat list
            predicted_diacritics = [id2diacritic[pid] for pid in prediction_ids]
        
        # 5. Reconstruct
        result = reconstruct_text(text, predicted_diacritics)
        
        return {"diacritized_text": result}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
