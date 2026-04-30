import json
import os

import matplotlib.pyplot as plt
import pandas as pd


try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
else:
    BASE_PATH = "."

LSTM_DIR = os.path.join(BASE_PATH, "file_train", "ltsm", "lstm")
PHOBERT_DIR = os.path.join(BASE_PATH, "file_train", "phobert")
VISUAL_DIR = os.path.join(BASE_PATH, "file_train", "visual", "visual")
os.makedirs(VISUAL_DIR, exist_ok=True)


def read_json(path):
    if not os.path.exists(path):
        print(f"[!] Missing history file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_val_f1(history):
    values = history.get("val_history", {}).get("f1", [])
    return max(values) if values else float(history.get("best_val_f1", history.get("final_val_f1", 0.0)))


def select_top_histories(histories, max_runs=6):
    return sorted(histories, key=best_val_f1, reverse=True)[:max_runs]


def make_label(result):
    dropout = result.get("dropout")
    batch_size = result.get("batch_size", result.get("BatchSize"))
    learning_rate = result.get("learning_rate", result.get("lr", result.get("LearningRate")))
    if learning_rate is None:
        return f"DR={dropout}, BS={batch_size}"
    return f"DR={dropout}, BS={batch_size}, LR={learning_rate:g}"


def has_nonzero_signal(values):
    return bool(values) and any(abs(float(value)) > 1e-12 for value in values)


def plot_history(histories, title, prefix, max_runs=6):
    if not histories:
        print(f"[!] No histories available for {title}.")
        return

    selected = select_top_histories(histories, max_runs=max_runs)
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    for result in selected:
        label = make_label(result)
        train_loss = result.get("train_history", {}).get("loss", [])
        val_loss = result.get("val_history", {}).get("loss", [])
        if train_loss:
            plt.plot(train_loss, label=f"Train {label}")
        if val_loss:
            plt.plot(val_loss, "--", label=f"Val {label}")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch / logged step")
    plt.ylabel("Loss")
    plt.legend(fontsize="x-small", ncol=1)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for result in selected:
        label = make_label(result)
        train_f1 = result.get("train_history", {}).get("f1", [])
        val_f1 = result.get("val_history", {}).get("f1", [])
        if has_nonzero_signal(train_f1):
            plt.plot(train_f1, label=f"Train {label}")
        if val_f1:
            plt.plot(val_f1, "--", label=f"Val {label}")
    plt.title(f"{title} - F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0, 1.0)
    plt.legend(fontsize="x-small", ncol=1)
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Top {len(selected)} runs by validation F1", y=1.02, fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(VISUAL_DIR, f"{prefix}_history.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def read_comparison(path):
    if not os.path.exists(path):
        print(f"[!] Missing comparison CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_model_comparison():
    lstm_comp = read_comparison(os.path.join(VISUAL_DIR, "lstm_comparison.csv"))
    phobert_comp = read_comparison(os.path.join(VISUAL_DIR, "phobert_comparison.csv"))
    if lstm_comp.empty or phobert_comp.empty:
        print("[!] Skip model comparison because one comparison CSV is missing.")
        return

    best_lstm_row = lstm_comp.loc[lstm_comp["Val_F1"].idxmax()]
    best_phobert_row = phobert_comp.loc[phobert_comp["Val_F1"].idxmax()]

    models = ["LSTM", "PhoBERT"]
    scores = [best_lstm_row["Val_F1"], best_phobert_row["Val_F1"]]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, scores, color=["#3498db", "#e74c3c"], alpha=0.85)
    plt.ylim(0, 1.0)
    plt.title("Best Model Comparison (F1 Score)", fontsize=14)
    plt.ylabel("F1 Score")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    output_path = os.path.join(VISUAL_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    lstm_histories = read_json(os.path.join(LSTM_DIR, "lstm_histories.json"))
    phobert_histories = read_json(os.path.join(PHOBERT_DIR, "phobert_histories.json"))

    plot_history(lstm_histories, "LSTM Training History", "lstm")
    plot_history(phobert_histories, "PhoBERT Training History", "phobert")
    plot_model_comparison()


if __name__ == "__main__":
    main()
