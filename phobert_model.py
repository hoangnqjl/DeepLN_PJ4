import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
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

PHOBERT_DIR = os.path.join(BASE_PATH, "phobert")
VISUAL_DIR = os.path.join(BASE_PATH, "visual")
os.makedirs(PHOBERT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
PHOBERT_BEST_PATH = os.path.join(PHOBERT_DIR, "phobert_best")
# ---------------------------------

class FakeNewsBERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

def run_phobert_experiment(dropout, batch_size, learning_rate, train_texts, train_labels, val_texts, val_labels):
    print(f"\n--- Starting PhoBERT Experiment: Dropout={dropout}, BatchSize={batch_size}, LR={learning_rate} ---")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=96)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=96)
    
    train_dataset = FakeNewsBERTDataset(train_encodings, train_labels)
    val_dataset = FakeNewsBERTDataset(val_encodings, val_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)
    
    # Freeze PhoBERT encoder layers to speed up training dramatically (80% faster)
    for param in model.roberta.parameters():
        param.requires_grad = False
        
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, f"phobert_dr{dropout}_bs{batch_size}_lr{learning_rate}"),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=os.path.join(BASE_PATH, 'logs'),
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    # Save model and tokenizer in format compatible with from_pretrained
    model_save_path = PHOBERT_BEST_PATH
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path) # Save tokenizer too for easy loading
    
    # Also save a .pth for compatibility if needed
    torch.save(model.state_dict(), os.path.join(model_save_path, "phobert_best.pth"))
    
    # Extract history
    history = trainer.state.log_history
    train_history = {'loss': [], 'f1': []}
    val_history = {'loss': [], 'f1': []}
    
    for log in history:
        if 'loss' in log and 'epoch' in log:
            train_history['loss'].append(log['loss'])
        if 'eval_loss' in log:
            val_history['loss'].append(log['eval_loss'])
        if 'eval_f1' in log:
            val_history['f1'].append(log['eval_f1'])
    
    # Fill in dummy F1 for training if not calculated to keep lengths consistent for plotting
    train_history['f1'] = [0.0] * len(train_history['loss'])
    
    return {
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_f1': eval_result['eval_f1'],
        'val_acc': eval_result.get('eval_accuracy', 0.0),
        'val_precision': eval_result.get('eval_precision', 0.0),
        'val_recall': eval_result.get('eval_recall', 0.0),
        'train_history': train_history,
        'val_history': val_history
    }

if __name__ == "__main__":
    # Load data from local or Drive
    train_path = os.path.join(BASE_PATH, "processed_data/train.csv")
    val_path = os.path.join(BASE_PATH, "processed_data/val.csv")
    
    if not os.path.exists(train_path):
        # Fallback to current directory if not in Drive relative path
        train_path = "processed_data/train.csv"
        val_path = "processed_data/val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Due to time constraints in demo, I will only run a few key combinations for PhoBERT
    # or just one if it's too slow.
    lrs = [2e-4] 
    dropouts = [0.1, 0.3, 0.5]
    batch_sizes = [8, 16, 32]
    
    all_bert_results = []
    
    for lr in lrs:
        for dr in dropouts:
            for bs in batch_sizes:
                res = run_phobert_experiment(dr, bs, lr, train_df['clean_message'], train_df['label'], val_df['clean_message'], val_df['label'])
                all_bert_results.append(res)
                
    report_df = pd.DataFrame([{
        'Dropout': r['dropout'],
        'BatchSize': r['batch_size'],
        'LearningRate': r['learning_rate'],
        'Val_F1': r['val_f1'],
        'Val_Acc': r['val_acc'],
        'Val_Precision': r['val_precision'],
        'Val_Recall': r['val_recall']
    } for r in all_bert_results])
    
    report_df.to_csv(os.path.join(VISUAL_DIR, "phobert_comparison.csv"), index=False)
    
    # Save all histories to a JSON for visualization (similar to LSTM)
    import json
    with open(os.path.join(PHOBERT_DIR, "phobert_histories.json"), "w") as f:
        json.dump(all_bert_results, f)
        
    print("\nPhoBERT Optimization Summary:")
    print(report_df)
