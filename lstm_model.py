import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import copy
import json
import os
import pickle
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Colab Support & Path Setup ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab.")
    # Check if drive is already mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("Google Drive is already mounted.")
        BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
    else:
        from google.colab import drive
        try:
            print("Attempting to mount Google Drive...")
            drive.mount('/content/drive')
            BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
        except Exception as e:
            print(f"\n[!] KHÔNG THỂ kết nối Drive tự động: {e}")
            print(" -> Nếu bạn chạy qua Terminal, hãy mount Drive bằng cell Notebook trước.")
            print(" -> Tiếp tục với bộ nhớ tạm thời của Colab...")
            BASE_PATH = "."

    if BASE_PATH != "." and not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Created project folder in Drive: {BASE_PATH}")
else:
    BASE_PATH = "."

LSTM_DIR = os.path.join(BASE_PATH, "lstm")
VISUAL_DIR = os.path.join(BASE_PATH, "visual")
os.makedirs(LSTM_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
# ---------------------------------

LABEL_MAPPING = {0: "Real", 1: "Fake"}
LSTM_ARCHITECTURE = {
    "embedding_dim": 100,
    "hidden_dim": 128,
    "output_dim": 2,
    "n_layers": 2,
    "max_len": 100,
}


def lr_to_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).split()
        indexed = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in text]
        
        if len(indexed) < self.max_len:
            indexed = [self.word_to_idx['<PAD>']] * (self.max_len - len(indexed)) + indexed
        else:
            indexed = indexed[:self.max_len]
            
        return torch.tensor(indexed), torch.tensor(self.labels[idx], dtype=torch.long)

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
        # Use the last hidden state
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_history = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
    val_history = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
    best_epoch = 0
    best_val_f1 = -1.0
    best_state_dict = None
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        train_history['loss'].append(total_loss/len(train_loader))
        train_history['f1'].append(train_f1)
        train_history['acc'].append(train_acc)
        train_history['precision'].append(train_prec)
        train_history['recall'].append(train_rec)
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        
        val_history['loss'].append(val_loss/len(val_loader))
        val_history['f1'].append(val_f1)
        val_history['acc'].append(val_acc)
        val_history['precision'].append(val_prec)
        val_history['recall'].append(val_rec)

        if val_f1 > best_val_f1:
            best_epoch = epoch + 1
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_metrics = {
                "val_loss": val_loss / len(val_loader),
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f} Acc: {val_acc:.4f}")
        
    return train_history, val_history, best_epoch, best_val_f1, best_state_dict, best_metrics

def run_experiment(dropout, batch_size, lr, train_texts, train_labels, val_texts, val_labels, word_to_idx, vocab_size):
    print(f"\n--- Starting Experiment: Dropout={dropout}, BatchSize={batch_size}, LR={lr} ---")
    
    train_dataset = FakeNewsDataset(train_texts, train_labels, word_to_idx)
    val_dataset = FakeNewsDataset(val_texts, val_labels, word_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Calculate class weights for imbalance (0: Real, 1: Fake)
    labels_count = Counter(train_labels)
    total_samples = len(train_labels)
    # Weight = total / (num_classes * count)
    weight_0 = total_samples / (2 * labels_count[0])
    weight_1 = total_samples / (2 * labels_count[1])
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
    print(f"Applying class weights: Real={weight_0:.2f}, Fake={weight_1:.2f}")

    model = FakeNewsLSTM(
        vocab_size,
        embedding_dim=LSTM_ARCHITECTURE["embedding_dim"],
        hidden_dim=LSTM_ARCHITECTURE["hidden_dim"],
        output_dim=LSTM_ARCHITECTURE["output_dim"],
        n_layers=LSTM_ARCHITECTURE["n_layers"],
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    train_hist, val_hist, best_epoch, best_val_f1, best_state_dict, best_metrics = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=15,
    )
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model_file = f"lstm_dr{dropout}_bs{batch_size}_lr{lr_to_tag(lr)}.pth"
    results_path = os.path.join(LSTM_DIR, model_file)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            **LSTM_ARCHITECTURE,
            "dropout": float(dropout),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "vocab_size": int(vocab_size),
            "label_mapping": LABEL_MAPPING,
        },
        "metrics": {
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_acc": float(best_metrics["val_acc"]),
            "best_val_precision": float(best_metrics["val_precision"]),
            "best_val_recall": float(best_metrics["val_recall"]),
            "final_val_f1": float(val_hist['f1'][-1]),
            "final_val_acc": float(val_hist['acc'][-1]),
            "final_val_precision": float(val_hist['precision'][-1]),
            "final_val_recall": float(val_hist['recall'][-1]),
        },
    }, results_path)
    
    return {
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': lr,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_metrics["val_acc"],
        'best_val_precision': best_metrics["val_precision"],
        'best_val_recall': best_metrics["val_recall"],
        'final_val_f1': val_hist['f1'][-1],
        'final_val_acc': val_hist['acc'][-1],
        'final_val_precision': val_hist['precision'][-1],
        'final_val_recall': val_hist['recall'][-1],
        'model_file': model_file,
        'model_path': results_path,
        'train_history': train_hist,
        'val_history': val_hist
    }

if __name__ == "__main__":
    # Load data from local or Drive
    train_path = os.path.join(BASE_PATH, "processed_data/train.csv")
    val_path = os.path.join(BASE_PATH, "processed_data/val.csv")
    
    if not os.path.exists(train_path):
        train_path = "processed_data/train.csv"
        val_path = "processed_data/val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Build vocab efficiently
    vocab = Counter()
    for msg in train_df['tokenized_message']:
        vocab.update(str(msg).split())
        
    most_common = vocab.most_common(10000)
    word_to_idx = {word: i+2 for i, (word, count) in enumerate(most_common)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    vocab_size = len(word_to_idx)
    
    train_texts = train_df['tokenized_message'].values
    train_labels = train_df['label'].values
    val_texts = val_df['tokenized_message'].values
    val_labels = val_df['label'].values
    
    lrs = [1e-3, 5e-4, 1e-4]
    dropouts = [0.1, 0.3, 0.5]
    batch_sizes = [8, 16, 32]
    
    all_results = []
    
    for lr in lrs:
        for dr in dropouts:
            for bs in batch_sizes:
                res = run_experiment(dr, bs, lr, train_texts, train_labels, val_texts, val_labels, word_to_idx, vocab_size)
                all_results.append(res)
            
    # Save comparison report and detailed histories
    report_df = pd.DataFrame([{ 
        'Dropout': r['dropout'], 
        'BatchSize': r['batch_size'],
        'LearningRate': r['learning_rate'],
        'BestEpoch': r['best_epoch'],
        'Val_F1': r['best_val_f1'],
        'Final_Val_F1': r['final_val_f1'],
        'Val_Acc': r['best_val_acc'],
        'Val_Precision': r['best_val_precision'],
        'Val_Recall': r['best_val_recall'],
        'ModelFile': r['model_file'],
    } for r in all_results])
    
    report_df.to_csv(os.path.join(VISUAL_DIR, "lstm_comparison.csv"), index=False)
    
    # Save all histories to a JSON for visualization
    with open(os.path.join(LSTM_DIR, "lstm_histories.json"), "w") as f:
        # Convert numpy floats to standard floats for JSON
        serializable_results = []
        for r in all_results:
            serializable_results.append({
                'dropout': float(r['dropout']),
                'batch_size': int(r['batch_size']),
                'learning_rate': float(r['learning_rate']),
                'model_file': r['model_file'],
                'best_epoch': int(r['best_epoch']),
                'best_val_f1': float(r['best_val_f1']),
                'best_val_acc': float(r['best_val_acc']),
                'best_val_precision': float(r['best_val_precision']),
                'best_val_recall': float(r['best_val_recall']),
                'final_val_f1': float(r['final_val_f1']),
                'train_history': {k: [float(x) for x in v] for k, v in r['train_history'].items()},
                'val_history': {k: [float(x) for x in v] for k, v in r['val_history'].items()}
            })
        json.dump(serializable_results, f)
        
    print("\nLSTM Optimization Summary:")
    print(report_df)
    
    # Save vocab for demo
    with open(os.path.join(LSTM_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)

    best_result = max(all_results, key=lambda r: r['best_val_f1'])
    best_model_path = os.path.join(LSTM_DIR, "lstm_best.pth")
    shutil.copyfile(best_result['model_path'], best_model_path)
    best_meta = {
        "model_file": "lstm_best.pth",
        "source_model_file": best_result["model_file"],
        "dropout": float(best_result["dropout"]),
        "batch_size": int(best_result["batch_size"]),
        "learning_rate": float(best_result["learning_rate"]),
        "best_epoch": int(best_result["best_epoch"]),
        "metrics": {
            "Val_F1": float(best_result["best_val_f1"]),
            "Val_Acc": float(best_result["best_val_acc"]),
            "Val_Precision": float(best_result["best_val_precision"]),
            "Val_Recall": float(best_result["best_val_recall"]),
        },
        "architecture": {
            **LSTM_ARCHITECTURE,
            "dropout": float(best_result["dropout"]),
            "vocab_size": int(vocab_size),
        },
        "label_mapping": LABEL_MAPPING,
    }
    with open(os.path.join(LSTM_DIR, "lstm_best_meta.json"), "w") as f:
        json.dump(best_meta, f, indent=2)

    print("\nBest LSTM model saved:")
    print(best_meta)
