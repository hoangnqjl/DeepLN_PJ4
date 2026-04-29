import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import inspect
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers.trainer_utils import get_last_checkpoint

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
RESULTS_DIR = os.path.join(PHOBERT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------

LABEL_MAPPING = {0: "Real", 1: "Fake"}
REQUIRED_RESULT_KEYS = {
    "dropout",
    "batch_size",
    "learning_rate",
    "run_name",
    "val_f1",
    "val_acc",
    "val_precision",
    "val_recall",
}


def lr_to_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")


def save_json_atomic(data, path, encoder=None):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=encoder)
    os.replace(tmp_path, path)


def load_valid_experiment_result(path, expected_run_name, dropout, batch_size, learning_rate):
    with open(path, "r", encoding="utf-8") as f:
        res = json.load(f)

    missing_keys = REQUIRED_RESULT_KEYS - set(res)
    if missing_keys:
        raise ValueError(f"missing keys: {sorted(missing_keys)}")

    if res["run_name"] != expected_run_name:
        raise ValueError(f"run_name mismatch: {res['run_name']} != {expected_run_name}")

    if abs(float(res["dropout"]) - float(dropout)) > 1e-12:
        raise ValueError("dropout mismatch")
    if int(res["batch_size"]) != int(batch_size):
        raise ValueError("batch_size mismatch")
    if abs(float(res["learning_rate"]) - float(learning_rate)) > 1e-12:
        raise ValueError("learning_rate mismatch")

    res["val_f1"] = float(res["val_f1"])
    return res


def build_training_args(**kwargs):
    strategy_key = "eval_strategy"
    if strategy_key not in inspect.signature(TrainingArguments.__init__).parameters:
        strategy_key = "evaluation_strategy"
    kwargs[strategy_key] = "epoch"
    return TrainingArguments(**kwargs)


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

def run_phobert_experiment(dropout, batch_size, learning_rate, train_texts, train_labels, val_texts, val_labels, best_score=-1.0):
    print(f"\n--- Starting PhoBERT Experiment: Dropout={dropout}, BatchSize={batch_size}, LR={learning_rate} ---")
    run_name = f"phobert_dr{dropout}_bs{batch_size}_lr{lr_to_tag(learning_rate)}"
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=96)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=96)
    
    train_dataset = FakeNewsBERTDataset(train_encodings, train_labels)
    val_dataset = FakeNewsBERTDataset(val_encodings, val_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    model.config.id2label = LABEL_MAPPING
    model.config.label2id = {"Real": 0, "Fake": 1}
    
    # Unfrozen PhoBERT encoder layers for full fine-tuning
    # for param in model.roberta.parameters():
    #     param.requires_grad = False

    model.to(device)
    
    training_args = build_training_args(
        output_dir=os.path.join(RESULTS_DIR, run_name),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=os.path.join(BASE_PATH, 'logs'),
        logging_steps=50,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming {run_name} from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    eval_result = trainer.evaluate()
    val_f1 = eval_result['eval_f1']
    saved_as_best = val_f1 > best_score
    
    if saved_as_best:
        os.makedirs(PHOBERT_BEST_PATH, exist_ok=True)
        trainer.save_model(PHOBERT_BEST_PATH)
        tokenizer.save_pretrained(PHOBERT_BEST_PATH)
        torch.save(model.state_dict(), os.path.join(PHOBERT_BEST_PATH, "phobert_best.pth"))

        best_meta = {
            "model_dir": "phobert_best",
            "source_run": run_name,
            "dropout": float(dropout),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "metrics": {
                "Val_F1": float(val_f1),
                "Val_Acc": float(eval_result.get('eval_accuracy', 0.0)),
                "Val_Precision": float(eval_result.get('eval_precision', 0.0)),
                "Val_Recall": float(eval_result.get('eval_recall', 0.0)),
            },
            "label_mapping": LABEL_MAPPING,
        }
        save_json_atomic(best_meta, os.path.join(PHOBERT_DIR, "phobert_best_meta.json"))
        print(f"New best PhoBERT saved to {PHOBERT_BEST_PATH} (F1={val_f1:.4f})")
    
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
    
    res_data = {
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'run_name': run_name,
        'saved_as_best': saved_as_best,
        'val_f1': val_f1,
        'val_acc': eval_result.get('eval_accuracy', 0.0),
        'val_precision': eval_result.get('eval_precision', 0.0),
        'val_recall': eval_result.get('eval_recall', 0.0),
        'train_history': train_history,
        'val_history': val_history
    }
    
    # Save individual experiment results to skip if rerun
    exp_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_json_atomic(res_data, os.path.join(exp_dir, "experiment_results.json"))
        
    return res_data

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
    
    lrs = [2e-5, 3e-5, 5e-5]
    dropouts = [0.1, 0.3, 0.5]
    batch_sizes = [8, 16, 32]


    
    all_bert_results = []
    best_score = -1.0
    
    for lr in lrs:
        for dr in dropouts:
            for bs in batch_sizes:
                run_name = f"phobert_dr{dr}_bs{bs}_lr{lr_to_tag(lr)}"
                exp_res_path = os.path.join(RESULTS_DIR, run_name, "experiment_results.json")
                
                if os.path.exists(exp_res_path):
                    print(f"\n>>> Skipping {run_name}, already trained. Loading results...")
                    try:
                        res = load_valid_experiment_result(exp_res_path, run_name, dr, bs, lr)
                        all_bert_results.append(res)
                        best_score = max(best_score, res['val_f1'])
                        continue
                    except Exception as e:
                        print(f"[!] Invalid result file for {run_name}: {e}. Retraining...")
                    
                res = run_phobert_experiment(
                    dr,
                    bs,
                    lr,
                    train_df['tokenized_message'],
                    train_df['label'],
                    val_df['tokenized_message'],
                    val_df['label'],
                    best_score=best_score,
                )
                all_bert_results.append(res)
                if res['saved_as_best']:
                    best_score = res['val_f1']
                
    report_df = pd.DataFrame([{
        'Dropout': r['dropout'],
        'BatchSize': r['batch_size'],
        'LearningRate': r['learning_rate'],
        'Val_F1': r['val_f1'],
        'Val_Acc': r['val_acc'],
        'Val_Precision': r['val_precision'],
        'Val_Recall': r['val_recall'],
        'SavedAsBest': r['saved_as_best'],
        'RunName': r['run_name'],
    } for r in all_bert_results])
    
    report_df.to_csv(os.path.join(VISUAL_DIR, "phobert_comparison.csv"), index=False)
    
    # Save all histories to a JSON for visualization (similar to LSTM)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    save_json_atomic(all_bert_results, os.path.join(PHOBERT_DIR, "phobert_histories.json"), encoder=NpEncoder)

        
    print("\nPhoBERT Optimization Summary:")
    print(report_df)

    best_result = max(all_bert_results, key=lambda r: r['val_f1'])
    print("\nBest PhoBERT model:")
    print({
        "run_name": best_result["run_name"],
        "dropout": best_result["dropout"],
        "batch_size": best_result["batch_size"],
        "learning_rate": best_result["learning_rate"],
        "Val_F1": best_result["val_f1"],
        "Val_Acc": best_result["val_acc"],
        "Val_Precision": best_result["val_precision"],
        "Val_Recall": best_result["val_recall"],
    })
