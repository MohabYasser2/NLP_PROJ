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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.arabert_char_bilstm_crf import AraBERTCharBiLSTMCRF
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
model = None
vocab = None
diacritic2id = None
id2diacritic = None
embedder = None
device = None

class DiacritizeRequest(BaseModel):
    text: str

def load_resources():
    global model, vocab, diacritic2id, id2diacritic, embedder, device
    
    print("Loading resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load diacritic mapping
    with open(os.path.join(BASE_DIR, "utils/diacritic2id.pickle"), "rb") as f:
        diacritic2id = pickle.load(f)
    id2diacritic = {v: k for k, v in diacritic2id.items()}

    # Load checkpoint
    model_path = os.path.join(BASE_DIR, "models/best_arabert_char_bilstm_crf (97.77).pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Rebuild vocab
    vocab = CharVocab()
    vocab.char2id = checkpoint['vocab']
    vocab.id2char = {v: k for k, v in vocab.char2id.items()}

    # Load config
    config = checkpoint['config']
    
    # Initialize embedder
    print("Initializing AraBERT embedder...")
    embedder = ContextualEmbedder(
        model_name="aubmindlab/bert-base-arabertv02",
        device=device.type,
        cache_dir=None
    )

    # Initialize model
    print("Initializing model...")
    model = AraBERTCharBiLSTMCRF(
        char_vocab_size=len(vocab.char2id),
        tagset_size=len(diacritic2id),
        arabert_dim=config['arabert_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Fix CRF parameter names (old checkpoint uses different names)
    state_dict = checkpoint['model_state_dict']
    if 'crf.trans_matrix' in state_dict:
        state_dict['crf.transitions'] = state_dict.pop('crf.trans_matrix')
    if 'crf.start_trans' in state_dict:
        state_dict['crf.start_transitions'] = state_dict.pop('crf.start_trans')
    if 'crf.end_trans' in state_dict:
        state_dict['crf.end_transitions'] = state_dict.pop('crf.end_trans')
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Resources loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

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
    text = request.text
    if not text:
        return {"diacritized_text": ""}

    try:
        # 1. Tokenize (to get base chars for model input)
        base_chars, _ = tokenize_line(text)
        
        if not base_chars:
             return {"diacritized_text": text}

        # 2. Prepare inputs
        # Char IDs
        char_ids = vocab.encode(base_chars)
        char_ids_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
        
        # AraBERT embeddings
        emb = embedder.embed_line_chars(text)
        # Convert numpy array to tensor efficiently
        import numpy as np
        emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
        emb_tensor = torch.from_numpy(emb_array).unsqueeze(0).float().to(device)
        
        # Mask
        mask = torch.ones((1, len(base_chars)), dtype=torch.bool).to(device)

        # 3. Inference
        with torch.no_grad():
            # Forward pass returns list of lists of tag IDs
            prediction_ids = model(emb_tensor, char_ids_tensor, mask=mask)
            
        # 4. Decode
        predicted_diacritics = [id2diacritic[pid] for pid in prediction_ids[0]]
        
        # 5. Reconstruct
        result = reconstruct_text(text, predicted_diacritics)
        
        return {"diacritized_text": result}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
