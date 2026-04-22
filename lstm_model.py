import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import os
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
    from google.colab import drive
    print("Detected Google Colab. Mounting Google Drive...")
    drive.mount('/content/drive')
    # Use a specific folder in Drive to persist results
    BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print(f"Working in Drive: {BASE_PATH}")
else:
    BASE_PATH = "."

RESULTS_DIR = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---------------------------------

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
            indexed += [self.word_to_idx['<PAD>']] * (self.max_len - len(indexed))
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
    train_history = {'loss': [], 'f1': []}
    val_history = {'loss': [], 'f1': []}
    
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
            
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_history['loss'].append(total_loss/len(train_loader))
        train_history['f1'].append(train_f1)
        
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
        
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_history['loss'].append(val_loss/len(val_loader))
        val_history['f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f}")
        
    return train_history, val_history

def run_experiment(dropout, batch_size, train_texts, train_labels, val_texts, val_labels, word_to_idx, vocab_size):
    print(f"\n--- Starting Experiment: Dropout={dropout}, BatchSize={batch_size} ---")
    
    train_dataset = FakeNewsDataset(train_texts, train_labels, word_to_idx)
    val_dataset = FakeNewsDataset(val_texts, val_labels, word_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = FakeNewsLSTM(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=2, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_hist, val_hist = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5)
    
    # Save the best model (using last as simplified version)
    results_path = os.path.join(RESULTS_DIR, f"lstm_dr{dropout}_bs{batch_size}.pth")
    torch.save(model.state_dict(), results_path)
    
    return {
        'dropout': dropout,
        'batch_size': batch_size,
        'final_val_f1': val_hist['f1'][-1],
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
    
    # Build vocab
    all_text = " ".join(train_df['tokenized_message'].tolist()).split()
    vocab = Counter(all_text)
    most_common = vocab.most_common(10000)
    word_to_idx = {word: i+2 for i, (word, count) in enumerate(most_common)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    vocab_size = len(word_to_idx)
    
    train_texts = train_df['tokenized_message'].values
    train_labels = train_df['label'].values
    val_texts = val_df['tokenized_message'].values
    val_labels = val_df['label'].values
    
    dropouts = [0.1, 0.3, 0.5]
    batch_sizes = [8, 16, 32]
    
    all_results = []
    
    for dr in dropouts:
        for bs in batch_sizes:
            res = run_experiment(dr, bs, train_texts, train_labels, val_texts, val_labels, word_to_idx, vocab_size)
            all_results.append(res)
            
    # Save comparison report and detailed histories
    report_df = pd.DataFrame([{ 
        'Dropout': r['dropout'], 
        'BatchSize': r['batch_size'], 
        'Val_F1': r['final_val_f1'] 
    } for r in all_results])
    
    report_df.to_csv(os.path.join(RESULTS_DIR, "lstm_comparison.csv"), index=False)
    
    # Save all histories to a JSON for visualization
    import json
    with open(os.path.join(RESULTS_DIR, "lstm_histories.json"), "w") as f:
        # Convert numpy floats to standard floats for JSON
        serializable_results = []
        for r in all_results:
            serializable_results.append({
                'dropout': float(r['dropout']),
                'batch_size': int(r['batch_size']),
                'final_val_f1': float(r['final_val_f1']),
                'train_history': {k: [float(x) for x in v] for k, v in r['train_history'].items()},
                'val_history': {k: [float(x) for x in v] for k, v in r['val_history'].items()}
            })
        json.dump(serializable_results, f)
        
    print("\nLSTM Optimization Summary:")
    print(report_df)
    
    # Save vocab for demo
    import pickle
    with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)
