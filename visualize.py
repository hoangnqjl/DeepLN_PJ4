import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Colab Support & Path Setup ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
else:
    BASE_PATH = "."

LSTM_DIR = os.path.join(BASE_PATH, "lstm")
PHOBERT_DIR = os.path.join(BASE_PATH, "phobert")
VISUAL_DIR = os.path.join(BASE_PATH, "visual")
os.makedirs(VISUAL_DIR, exist_ok=True)

def plot_history(histories, title, prefix):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for res in histories:
        label = f"DR={res['dropout']}, BS={res.get('batch_size', res.get('lr'))}"
        plt.plot(res['train_history']['loss'], label=f"Train {label}")
        plt.plot(res['val_history']['loss'], '--', label=f"Val {label}")
    plt.title(f"{title} - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Plot F1
    plt.subplot(1, 2, 2)
    for res in histories:
        label = f"DR={res['dropout']}, BS={res.get('batch_size', res.get('lr'))}"
        plt.plot(res['train_history']['f1'], label=f"Train {label}")
        plt.plot(res['val_history']['f1'], '--', label=f"Val {label}")
    plt.title(f"{title} - F1 Score")
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f'{prefix}_history.png'))
    print(f"Saved {prefix}_history.png to {VISUAL_DIR}")

def main():
    # Load LSTM histories
    with open(os.path.join(LSTM_DIR, 'lstm_histories.json'), 'r') as f:
        lstm_histories = json.load(f)
    
    with open(os.path.join(PHOBERT_DIR, 'phobert_histories.json'), 'r') as f:
        phobert_histories = json.load(f)
        
    # Filter for a subset of histories to make the plot readable (e.g., best ones or representative ones)
    # For simplicity, we'll plot all but maybe just the best few? No, let's plot all as requested.
    
    plot_history(lstm_histories, "LSTM Training History", "lstm")
    plot_history(phobert_histories, "PhoBERT Training History", "phobert")
    
    # Comparison Bar Chart
    lstm_comp = pd.read_csv(os.path.join(VISUAL_DIR, 'lstm_comparison.csv'))
    phobert_comp = pd.read_csv(os.path.join(VISUAL_DIR, 'phobert_comparison.csv'))
    
    best_lstm = lstm_comp['Val_F1'].max()
    best_phobert = phobert_comp['Val_F1'].max()
    
    plt.figure(figsize=(8, 6))
    models = ['LSTM', 'PhoBERT']
    scores = [best_lstm, best_phobert]
    bars = plt.bar(models, scores, color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.ylim(0, 1.0)
    plt.title('Best Model Comparison (F1 Score)', fontsize=14)
    plt.ylabel('F1 Score')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'model_comparison.png'))
    print("Saved model_comparison.png")

if __name__ == "__main__":
    main()
