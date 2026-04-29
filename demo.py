import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification
import pickle
import os
from pyvi import ViTokenizer
import re

# --- Colab Support & Path Setup ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab.")
    if os.path.exists('/content/drive/MyDrive'):
        BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
    else:
        from google.colab import drive
        try:
            drive.mount('/content/drive')
            BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
        except Exception as e:
            BASE_PATH = "."
else:
    BASE_PATH = "."

LSTM_DIR = os.path.join(BASE_PATH, "lstm")
PHOBERT_DIR = os.path.join(BASE_PATH, "phobert")
# ---------------------------------


def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- LSTM Model Definition (must match training) ---
class FakeNewsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(FakeNewsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)

def predict_lstm(text, model, word_to_idx, max_len=100):
    model.eval()
    cleaned = clean_text(text)
    tokenized = ViTokenizer.tokenize(cleaned).split()
    indexed = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokenized]
    if len(indexed) < max_len:
        indexed = [0] * (max_len - len(indexed)) + indexed
    else:
        indexed = indexed[:max_len]
    
    tensor = torch.tensor([indexed]).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
    return "Real" if prediction == 0 else "Fake"

def predict_phobert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Real" if prediction == 0 else "Fake"

def main():
    print("--- Fake News Detection Demo ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load LSTM
    with open(os.path.join(LSTM_DIR, 'vocab.pkl'), 'rb') as f:
        word_to_idx = pickle.load(f)
    
    lstm_model = FakeNewsLSTM(len(word_to_idx), 100, 128, 2, 2, 0.5).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(LSTM_DIR, 'lstm_dr0.5_bs32.pth'), map_location=device))
    
    # 2. Load PhoBERT
    phobert_path = os.path.join(PHOBERT_DIR, 'phobert_best')

    
    if not os.path.exists(phobert_path):
        print(f"\n[!] LỖI: Không tìm thấy mô hình PhoBERT tại '{phobert_path}'")
        print(" -> Nếu bạn vừa train trên Colab, hãy tải thư mục 'results/phobert_best' về máy.")
        print(" -> Nếu chưa train, hãy chạy 'python phobert_model.py' để tạo mô hình.")
        phobert_model = None
        phobert_tokenizer = None
    else:
        print("Đang tải mô hình PhoBERT...")
        phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_path)
        phobert_model = RobertaForSequenceClassification.from_pretrained(phobert_path).to(device)
    
    while True:
        print("\n" + "="*30)
        text = input("Nhập văn bản tin tức (hoặc 'q' để thoát): ")
        if text.lower() == 'q':
            break
            
        print(f"\nKết quả dự đoán:")
        print(f" - LSTM:    {predict_lstm(text, lstm_model, word_to_idx)}")
        if phobert_model:
            print(f" - PhoBERT: {predict_phobert(text, phobert_model, phobert_tokenizer)}")
        else:
            print(f" - PhoBERT: (Chưa tải được mô hình)")

if __name__ == "__main__":
    main()
