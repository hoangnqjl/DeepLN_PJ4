import os

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

VISUAL_DIR = os.path.join(BASE_PATH, "file_train", "visual", "visual")


def read_comparison(filename):
    path = os.path.join(VISUAL_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] Missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def format_best_params(row):
    parts = [
        f"DR={row.get('Dropout', '-')}",
        f"BS={int(row['BatchSize']) if 'BatchSize' in row and pd.notna(row['BatchSize']) else '-'}",
    ]
    if "LearningRate" in row and pd.notna(row["LearningRate"]):
        parts.append(f"LR={row['LearningRate']}")
    return ", ".join(parts)


def summarize_search_space(name, df):
    if df.empty:
        print(f"{name}: no data.")
        return

    dropouts = sorted(df["Dropout"].dropna().unique().tolist()) if "Dropout" in df else []
    batch_sizes = sorted(df["BatchSize"].dropna().unique().tolist()) if "BatchSize" in df else []
    learning_rates = sorted(df["LearningRate"].dropna().unique().tolist()) if "LearningRate" in df else []

    print(f"{name} search space:")
    print(f" - Runs: {len(df)}")
    print(f" - Dropout: {dropouts}")
    print(f" - Batch size: {batch_sizes}")
    print(f" - Learning rate: {learning_rates if learning_rates else 'MISSING'}")


def print_top_results(name, df, top_n=10):
    print(f"\n## {name} Model Experiments")
    if df.empty:
        print("No data found.")
        return None

    sorted_df = df.sort_values(by="Val_F1", ascending=False).head(top_n)
    columns = [
        column for column in [
            "Dropout",
            "BatchSize",
            "LearningRate",
            "BestEpoch",
            "Val_F1",
            "Val_Acc",
            "Val_Precision",
            "Val_Recall",
            "ModelFile",
            "RunName",
        ] if column in sorted_df.columns
    ]
    print(sorted_df[columns].to_string(index=False))
    return df.loc[df["Val_F1"].idxmax()]


def main():
    lstm_comp = read_comparison("lstm_comparison.csv")
    phobert_comp = read_comparison("phobert_comparison.csv")

    print("# Final Evaluation Report - Fake News Detection")
    print("\nLabel mapping: 0 = Real, 1 = Fake")
    print("\n## Hyperparameter Coverage")
    summarize_search_space("LSTM", lstm_comp)
    summarize_search_space("PhoBERT", phobert_comp)

    best_lstm = print_top_results("LSTM", lstm_comp)
    best_phobert = print_top_results("PhoBERT", phobert_comp)

    print("\n## Best Model Summary")
    summary = []
    if best_lstm is not None:
        summary.append({
            "Model": "LSTM",
            "Best Params": format_best_params(best_lstm),
            "Accuracy": best_lstm.get("Val_Acc", "-"),
            "Precision": best_lstm.get("Val_Precision", "-"),
            "Recall": best_lstm.get("Val_Recall", "-"),
            "F1": best_lstm["Val_F1"],
        })
    if best_phobert is not None:
        summary.append({
            "Model": "PhoBERT",
            "Best Params": format_best_params(best_phobert),
            "Accuracy": best_phobert.get("Val_Acc", "-"),
            "Precision": best_phobert.get("Val_Precision", "-"),
            "Recall": best_phobert.get("Val_Recall", "-"),
            "F1": best_phobert["Val_F1"],
        })

    if summary:
        print(pd.DataFrame(summary).to_string(index=False))

    print("\n## Conclusion")
    if best_lstm is not None and best_phobert is not None:
        diff = float(best_phobert["Val_F1"]) - float(best_lstm["Val_F1"])
        if diff > 0:
            print(f"PhoBERT outperformed LSTM by {diff:.4f} F1-score.")
        elif diff < 0:
            print(f"LSTM outperformed PhoBERT by {-diff:.4f} F1-score.")
        else:
            print("LSTM and PhoBERT reached the same F1-score.")
    else:
        print("Cannot compare models because one result table is missing.")


if __name__ == "__main__":
    main()
