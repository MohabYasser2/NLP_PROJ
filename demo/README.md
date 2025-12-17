# Arabic Diacritization Demo

A terminal-styled full-stack demo for Arabic text diacritization using the trained AraBERT + Character Fusion BiLSTM-CRF model.

## Backend (FastAPI)

### Features

- Loads the trained `best_arabert_char_bilstm_crf.pth` model
- Uses the same preprocessing, tokenization, and inference logic as `test.py`
- Single POST endpoint `/diacritize` that accepts non-diacritized Arabic text
- Returns fully diacritized text

### Running the Backend

From the project root directory:

```bash
cd c:\Users\Aisha\Desktop\NLP_PROJ
python -m uvicorn demo.app:app --reload
```

The server will start at `http://127.0.0.1:8000`

### API Endpoint

**POST** `/diacritize`

**Request Body:**

```json
{
  "text": "السلام عليكم"
}
```

**Response:**

```json
{
  "diacritized_text": "السَّلَامُ عَلَيْكُمْ"
}
```

## Frontend (HTML + CSS + JavaScript)

### Features

- Terminal-style UI (black background, green text)
- Monospace font for authentic terminal feel
- Input text area for non-diacritized Arabic
- "RUN_EXECUTE" button to process text
- Output area with typewriter effect
- Blinking cursor animation

### Running the Frontend

Simply open `index.html` in a web browser:

```bash
# Option 1: Double-click index.html
# Option 2: Open in browser directly
start demo/index.html
```

**Note:** The backend must be running at `http://localhost:8000` for the frontend to work.

## Usage

1. Start the backend server (see above)
2. Open `index.html` in your browser
3. Enter Arabic text without diacritics in the text box
4. Click "RUN_EXECUTE"
5. View the diacritized output with terminal typewriter effect

## Example

**Input:**

```
قال ابو بكر
```

**Output:**

```
قَالَ أَبُو بَكْرٍ
```

## Technical Details

- **Model:** AraBERT + Character Fusion BiLSTM-CRF
- **Preprocessing:** Identical to `test.py` (tokenization, embedding, encoding)
- **No database, no authentication** - pure demo
- **AraBERT embeddings computed on-the-fly**
- **CRF decoding for sequence prediction**

## Requirements

- Python 3.11+
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- TorchCRF
- All dependencies from `requirements.txt`
