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
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'f1': f1,
    }

def run_phobert_experiment(dropout, batch_size, learning_rate, train_texts, train_labels, val_texts, val_labels):
    print(f"\n--- Starting PhoBERT Experiment: Dropout={dropout}, BatchSize={batch_size}, LR={learning_rate} ---")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)
    
    train_dataset = FakeNewsBERTDataset(train_encodings, train_labels)
    val_dataset = FakeNewsBERTDataset(val_encodings, val_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=f"./results/phobert_dr{dropout}_bs{batch_size}_lr{learning_rate}",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs',
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
    
    # Save model
    model_save_path = f"d:/Data/VKU/Projects/DeepLN_PJ4/results/phobert_best.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    
    return {
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_f1': eval_result['eval_f1'],
    }

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("d:/Data/VKU/Projects/DeepLN_PJ4/processed_data/train.csv")
    val_df = pd.read_csv("d:/Data/VKU/Projects/DeepLN_PJ4/processed_data/val.csv")
    
    # Due to time constraints in demo, I will only run a few key combinations for PhoBERT
    # or just one if it's too slow.
    lrs = [5e-5] # Simplified for demo
    dropouts = [0.1]
    batch_sizes = [16]
    
    all_bert_results = []
    
    for lr in lrs:
        for dr in dropouts:
            for bs in batch_sizes:
                res = run_phobert_experiment(dr, bs, lr, train_df['clean_message'], train_df['label'], val_df['clean_message'], val_df['label'])
                all_bert_results.append(res)
                
    report_df = pd.DataFrame(all_bert_results)
    report_df.to_csv("d:/Data/VKU/Projects/DeepLN_PJ4/results/phobert_comparison.csv", index=False)
    print("\nPhoBERT Optimization Summary:")
    print(report_df)
