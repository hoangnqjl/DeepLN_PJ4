import os
import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

# Import từ các file đã có (chỉnh đường dẫn cho đúng cấu trúc mới)
from demo import load_lstm, load_phobert, predict_lstm, predict_phobert
from tools.vn_ocr import extract_text_from_image

app = FastAPI(title="Fake News Detection")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm_model, word_to_idx, lstm_max_len = None, None, 100
phobert_model, phobert_tokenizer = None, None

@app.on_event("startup")
async def load_models():
    global lstm_model, word_to_idx, lstm_max_len
    global phobert_model, phobert_tokenizer
    print("🚀 Đang tải các mô hình AI...")
    lstm_model, word_to_idx, lstm_max_len = load_lstm(device)
    phobert_model, phobert_tokenizer = load_phobert(device)
    print("✅ Mô hình đã sẵn sàng!")

# Mount static files (để chứa CSS/JS)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/text")
async def api_predict_text(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    
    res_lstm = predict_lstm(text, lstm_model, word_to_idx, lstm_max_len) if lstm_model else "N/A"
    res_phobert = predict_phobert(text, phobert_model, phobert_tokenizer) if phobert_model else "N/A"
    
    return {
        "text": text,
        "lstm": res_lstm,
        "phobert": res_phobert
    }

@app.post("/predict/image")
async def api_predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # OCR trích xuất text
    text = extract_text_from_image(img)
    
    if not text.strip():
        return {"text": "", "lstm": "N/A", "phobert": "N/A", "error": "Không tìm thấy văn bản trong ảnh"}

    # Dự đoán từ text OCR
    res_lstm = predict_lstm(text, lstm_model, word_to_idx, lstm_max_len) if lstm_model else "N/A"
    res_phobert = predict_phobert(text, phobert_model, phobert_tokenizer) if phobert_model else "N/A"
    
    return {
        "text": text,
        "lstm": res_lstm,
        "phobert": res_phobert
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
