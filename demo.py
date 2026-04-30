import csv
import json
import os
import pickle
import re

import torch
import torch.nn as nn
from pyvi import ViTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
        except Exception:
            BASE_PATH = "."
else:
    BASE_PATH = "."

LSTM_DIR = os.path.join(BASE_PATH, "file_train", "ltsm", "lstm")
PHOBERT_DIR = os.path.join(BASE_PATH, "file_train", "phobert")
VISUAL_DIR = os.path.join(BASE_PATH, "file_train", "visual", "visual")

LABEL_MAPPING = {0: "Real", 1: "Fake"}


def lr_to_tag(lr):
    return f"{float(lr):.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_best_row(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("Val_F1", -1.0)))


def first_existing(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


class FakeNewsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(FakeNewsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


def resolve_lstm_config():
    meta = load_json(os.path.join(LSTM_DIR, "lstm_best_meta.json"))
    default_arch = {
        "embedding_dim": 100,
        "hidden_dim": 128,
        "output_dim": 2,
        "n_layers": 2,
        "dropout": 0.5,
        "max_len": 100,
    }

    if meta:
        arch = {**default_arch, **meta.get("architecture", {})}
        model_file = meta.get("model_file", "lstm_best.pth")
        return arch, [
            os.path.join(LSTM_DIR, model_file),
            os.path.join(LSTM_DIR, meta.get("source_model_file", "")),
        ]

    best_row = read_best_row(os.path.join(VISUAL_DIR, "lstm_comparison.csv"))
    if best_row:
        dropout = float(best_row.get("Dropout", default_arch["dropout"]))
        batch_size = int(float(best_row.get("BatchSize", 32)))
        learning_rate = best_row.get("LearningRate")
        model_file = best_row.get("ModelFile")
        arch = {**default_arch, "dropout": dropout}
        candidates = []
        if model_file:
            candidates.append(os.path.join(LSTM_DIR, model_file))
        if learning_rate:
            candidates.append(os.path.join(LSTM_DIR, f"lstm_dr{dropout}_bs{batch_size}_lr{lr_to_tag(learning_rate)}.pth"))
        candidates.append(os.path.join(LSTM_DIR, f"lstm_dr{dropout}_bs{batch_size}.pth"))
        return arch, candidates

    return default_arch, [
        os.path.join(LSTM_DIR, "lstm_best.pth"),
        os.path.join(LSTM_DIR, "lstm_dr0.5_bs32.pth"),
    ]


def load_lstm(device):
    vocab_path = first_existing([
        os.path.join(LSTM_DIR, "vocab.pkl"),
        os.path.join(BASE_PATH, "results", "vocab.pkl"),
    ])
    if not vocab_path:
        print("[!] LSTM vocab not found. Skipping LSTM.")
        return None, None, 100

    with open(vocab_path, "rb") as f:
        word_to_idx = pickle.load(f)

    arch, candidates = resolve_lstm_config()
    model_path = first_existing(candidates)
    if not model_path:
        print("[!] LSTM model file not found. Skipping LSTM.")
        return None, word_to_idx, int(arch.get("max_len", 100))

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        arch.update({
            "embedding_dim": config.get("embedding_dim", arch["embedding_dim"]),
            "hidden_dim": config.get("hidden_dim", arch["hidden_dim"]),
            "output_dim": config.get("output_dim", arch["output_dim"]),
            "n_layers": config.get("n_layers", arch["n_layers"]),
            "dropout": config.get("dropout", arch["dropout"]),
            "max_len": config.get("max_len", arch["max_len"]),
        })
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = FakeNewsLSTM(
        len(word_to_idx),
        int(arch["embedding_dim"]),
        int(arch["hidden_dim"]),
        int(arch["output_dim"]),
        int(arch["n_layers"]),
        float(arch["dropout"]),
    ).to(device)
    model.load_state_dict(state_dict)
    print(f"Loaded LSTM from {model_path}")
    return model, word_to_idx, int(arch.get("max_len", 100))


def load_phobert(device):
    meta = load_json(os.path.join(PHOBERT_DIR, "phobert_best_meta.json"))
    model_dir = meta.get("model_dir", "phobert_best") if meta else "phobert_best"
    phobert_path = os.path.join(PHOBERT_DIR, model_dir)

    if not os.path.exists(phobert_path):
        print(f"[!] PhoBERT model not found at {phobert_path}. Skipping PhoBERT.")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(phobert_path)
    model = AutoModelForSequenceClassification.from_pretrained(phobert_path).to(device)
    print(f"Loaded PhoBERT from {phobert_path}")
    return model, tokenizer


def predict_lstm(text, model, word_to_idx, max_len):
    model.eval()
    cleaned = clean_text(text)
    tokenized = ViTokenizer.tokenize(cleaned).split()
    indexed = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokenized]
    if len(indexed) < max_len:
        indexed = [word_to_idx['<PAD>']] * (max_len - len(indexed)) + indexed
    else:
        indexed = indexed[:max_len]

    tensor = torch.tensor([indexed]).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
    return LABEL_MAPPING[prediction]


def predict_phobert(text, model, tokenizer):
    model.eval()
    cleaned = clean_text(text)
    tokenized = ViTokenizer.tokenize(cleaned)
    inputs = tokenizer(tokenized, return_tensors="pt", truncation=True, padding=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return LABEL_MAPPING[prediction]


def main():
    print("--- Fake News Detection Demo ---")
    print("Label mapping: 0 = Real, 1 = Fake")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("🚀 Đang chạy bằng GPU (CUDA)!")
    else:
        print("⚠️ Đang chạy bằng CPU!")
    lstm_model, word_to_idx, lstm_max_len = load_lstm(device)
    phobert_model, phobert_tokenizer = load_phobert(device)

    if lstm_model is None and phobert_model is None:
        print("[!] No trained model found. Please train LSTM and/or PhoBERT first.")
        return

    while True:
        print("\n" + "=" * 30)
        text = input("Nhap van ban tin tuc/post (hoac 'q' de thoat): ")
        if text.lower() == 'q':
            break

        print("\nKet qua du doan:")
        if lstm_model is not None:
            print(f" - LSTM:    {predict_lstm(text, lstm_model, word_to_idx, lstm_max_len)}")
        else:
            print(" - LSTM:    model not available")

        if phobert_model is not None:
            print(f" - PhoBERT: {predict_phobert(text, phobert_model, phobert_tokenizer)}")
        else:
            print(" - PhoBERT: model not available")


if __name__ == "__main__":
    main()
