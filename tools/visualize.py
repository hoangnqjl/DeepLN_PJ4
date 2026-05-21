import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = "/content/drive/MyDrive/DeepLN_PJ4"
else:
    BASE_PATH = "."

VISUAL_DIR = os.path.join(BASE_PATH, "checkpoints", "visual")
os.makedirs(VISUAL_DIR, exist_ok=True)


def locate_history(model_type, version):
    """
    Tìm kiếm file history JSON của mô hình trên cả Colab/Drive và Local.
    """
    candidates = []
    if version == 1:
        if model_type == "lstm":
            candidates.extend([
                "checkpoints/lstm_data_01",
                "lstm_data_01",
                "checkpoints/lstm_01",
                "lstm_01"
            ])
        else:
            candidates.extend([
                "checkpoints/checkpoints_01",
                "checkpoints_01",
                "checkpoints/phobert_01",
                "phobert_01"
            ])
    elif version == 2:
        if model_type == "lstm":
            candidates.extend([
                "checkpoints/checkpoints_02/lstm",
                "checkpoints_02/lstm",
                "checkpoints/lstm_data_02",
                "lstm_data_02"
            ])
        else:
            candidates.extend([
                "checkpoints/checkpoints_02/phobert",
                "checkpoints_02/phobert",
                "checkpoints/checkpoints_02",
                "checkpoints_02",
                "checkpoints/phobert_data_02",
                "phobert_data_02"
            ])
    elif version == 3:
        if model_type == "lstm":
            candidates.extend([
                "checkpoints/lstm_data_03",
                "lstm_data_03",
                "checkpoints/lstm_03",
                "lstm_03"
            ])
        else:
            candidates.extend([
                "checkpoints/checkpoints_03",
                "checkpoints_03",
                "checkpoints/phobert_03",
                "phobert_03"
            ])

    filename = "lstm_histories.json" if model_type == "lstm" else "phobert_histories.json"
    
    for c in candidates:
        for base in [BASE_PATH, "."]:
            path = os.path.join(base, c, filename)
            if os.path.exists(path):
                return path
            path_flat = os.path.join(base, c.replace("checkpoints/", ""), filename)
            if os.path.exists(path_flat):
                return path_flat
    return None


def locate_comparison(model_type, version):
    """
    Tìm kiếm file comparison CSV của mô hình.
    """
    candidates = []
    if version == 1:
        if model_type == "lstm":
            candidates.extend(["checkpoints/lstm_data_01", "lstm_data_01"])
        else:
            candidates.extend(["checkpoints/checkpoints_01", "checkpoints_01"])
    elif version == 2:
        candidates.extend(["checkpoints/checkpoints_02/visual", "checkpoints_02/visual"])
    elif version == 3:
        if model_type == "lstm":
            candidates.extend(["checkpoints/lstm_data_03", "lstm_data_03"])
        else:
            candidates.extend(["checkpoints/checkpoints_03", "checkpoints_03"])

    filename = f"{model_type}_comparison.csv"
    
    # Thử tìm theo cấu trúc thư mục
    for c in candidates:
        for base in [BASE_PATH, "."]:
            path = os.path.join(base, c, filename)
            if os.path.exists(path):
                return path
            path_flat = os.path.join(base, c.replace("checkpoints/", ""), filename)
            if os.path.exists(path_flat):
                return path_flat

    # Fallback tìm trong thư mục visual chung
    for base in [BASE_PATH, "."]:
        path = os.path.join(base, "checkpoints/visual", f"{model_type}_comparison_v{version}.csv")
        if os.path.exists(path):
            return path
        path_generic = os.path.join(base, "checkpoints/visual", filename)
        if os.path.exists(path_generic):
            # Cần chắc chắn đây không phải của version khác
            return path_generic
            
    return None


def read_json(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"[!] Error reading JSON {path}: {e}")
            return []


def best_val_f1(history):
    values = history.get("val_history", {}).get("f1", [])
    return max(values) if values else float(history.get("best_val_f1", history.get("final_val_f1", history.get("val_f1", 0.0))))


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


def reconstruct_comparison_df(model_type, histories):
    if not histories:
        return pd.DataFrame()
        
    records = []
    for r in histories:
        best_f1 = best_val_f1(r)
        dr = r.get("dropout", r.get("Dropout"))
        bs = r.get("batch_size", r.get("BatchSize"))
        lr = r.get("learning_rate", r.get("lr", r.get("LearningRate")))
        
        if model_type == "lstm":
            records.append({
                'Dropout': dr,
                'BatchSize': bs,
                'LearningRate': lr,
                'BestEpoch': r.get('best_epoch', r.get('BestEpoch', 0)),
                'Val_F1': best_f1,
                'Final_Val_F1': r.get('final_val_f1', r.get('Final_Val_F1', best_f1)),
                'Val_Acc': r.get('best_val_acc', r.get('Val_Acc', r.get('val_acc', 0.0))),
                'Val_Precision': r.get('best_val_precision', r.get('Val_Precision', r.get('val_precision', 0.0))),
                'Val_Recall': r.get('best_val_recall', r.get('Val_Recall', r.get('val_recall', 0.0))),
                'ModelFile': r.get('model_file', r.get('ModelFile', ''))
            })
        else:
            records.append({
                'Dropout': dr,
                'BatchSize': bs,
                'LearningRate': lr,
                'Val_F1': best_f1,
                'Val_Acc': r.get('val_acc', r.get('Val_Acc', 0.0)),
                'Val_Precision': r.get('val_precision', r.get('Val_Precision', 0.0)),
                'Val_Recall': r.get('val_recall', r.get('Val_Recall', 0.0)),
                'SavedAsBest': r.get('saved_as_best', r.get('SavedAsBest', False)),
                'RunName': r.get('run_name', r.get('RunName', ''))
            })
            
    return pd.DataFrame(records)


def get_comparison_df(model_type, version, histories):
    csv_path = locate_comparison(model_type, version)
    if csv_path:
        print(f"[+] Found {model_type} comparison CSV for V{version} at: {csv_path}")
        return pd.read_csv(csv_path)
    
    if histories:
        print(f"[!] {model_type} comparison CSV for V{version} not found. Reconstructing from histories JSON...")
        return reconstruct_comparison_df(model_type, histories)
        
    return pd.DataFrame()


def plot_history(histories, title, prefix, version, max_runs=6):
    if not histories:
        print(f"[!] No histories available for {title} (Version {version}).")
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
    plt.title(f"{title} (V{version}) - Loss")
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
    plt.title(f"{title} (V{version}) - F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0, 1.0)
    plt.legend(fontsize="x-small", ncol=1)
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Top {len(selected)} runs by validation F1", y=1.02, fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(VISUAL_DIR, f"{prefix}_v{version}_history.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_single_version_comparison(lstm_comp, phobert_comp, version):
    if lstm_comp.empty and phobert_comp.empty:
        print(f"[!] Skip model comparison for V{version} because both comparison dataframes are empty.")
        return None, None

    best_lstm_f1 = 0.0
    best_phobert_f1 = 0.0
    best_lstm_row = None
    best_phobert_row = None

    if not lstm_comp.empty and "Val_F1" in lstm_comp.columns:
        best_lstm_idx = lstm_comp["Val_F1"].idxmax()
        best_lstm_row = lstm_comp.loc[best_lstm_idx]
        best_lstm_f1 = best_lstm_row["Val_F1"]
        
    if not phobert_comp.empty and "Val_F1" in phobert_comp.columns:
        best_phobert_idx = phobert_comp["Val_F1"].idxmax()
        best_phobert_row = phobert_comp.loc[best_phobert_idx]
        best_phobert_f1 = best_phobert_row["Val_F1"]

    # Chỉ vẽ biểu đồ so sánh đơn lẻ nếu có đủ cả hai mô hình của phiên bản đó
    if best_lstm_row is not None and best_phobert_row is not None:
        models = ["LSTM", "PhoBERT"]
        scores = [best_lstm_f1, best_phobert_f1]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, scores, color=["#3498db", "#e74c3c"], alpha=0.85, edgecolor='black', linewidth=0.7)
        plt.ylim(0, 1.0)
        plt.title(f"Best Model Comparison - Version {version} (F1 Score)", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("F1 Score", fontsize=12)
        plt.grid(axis="y", alpha=0.3, linestyle="--")

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

        output_path = os.path.join(VISUAL_DIR, f"model_comparison_v{version}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

    return best_lstm_row, best_phobert_row


def plot_master_comparison(best_results):
    """
    Vẽ biểu đồ cột so sánh tổng hợp cả 3 phiên bản dữ liệu.
    """
    df = pd.DataFrame(best_results)
    if df.empty:
        print("[!] No comparison results available to plot master comparison.")
        return

    plt.figure(figsize=(12, 7))
    
    # Dữ liệu cột
    versions = ['V1', 'V2', 'V3']
    x = np.arange(len(versions))
    width = 0.35
    
    lstm_scores = []
    phobert_scores = []
    
    for v in versions:
        lstm_row = df[(df['Version'] == v) & (df['Model'] == 'LSTM')]
        phobert_row = df[(df['Version'] == v) & (df['Model'] == 'PhoBERT')]
        
        lstm_scores.append(lstm_row['Val_F1'].values[0] if not lstm_row.empty else 0.0)
        phobert_scores.append(phobert_row['Val_F1'].values[0] if not phobert_row.empty else 0.0)
    
    rects1 = plt.bar(x - width/2, lstm_scores, width, label='Bi-LSTM', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.7)
    rects2 = plt.bar(x + width/2, phobert_scores, width, label='PhoBERT', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.7)
    
    plt.xlabel('Dataset & Training Method', fontsize=12, fontweight='bold', labelpad=12)
    plt.ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
    plt.title('Comparative Analysis Across All 3 Model Versions (F1 Score)', fontsize=15, fontweight='bold', pad=20)
    
    # Định dạng nhãn trục hoành rõ ràng hơn
    plt.xticks(x, [
        'V1 (data_01 - 4.3K samples)\nPhoBERT Baseline vs LSTM',
        'V2 (data_02 - 4.9K samples)\nPhoBERT Baseline vs LSTM',
        'V3 (data_03 - 5.6K samples)\nPhoBERT LoRA vs LSTM'
    ], fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    # Đánh dấu nhãn số trên đỉnh các cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                plt.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),  # offset 4pt theo phương đứng
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    output_path = os.path.join(VISUAL_DIR, "all_versions_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Master Comparison to {output_path}")


def main():
    print("==================================================")
    print("[+] STARTING THREE-VERSION PERFORMANCE VISUALIZATION")
    print("==================================================")
    
    best_results_list = []
    
    for version in [1, 2, 3]:
        print(f"\nProcessing Version {version}...")
        
        # 1. Tìm và đọc files histories
        lstm_path = locate_history("lstm", version)
        phobert_path = locate_history("phobert", version)
        
        lstm_hist = read_json(lstm_path) if lstm_path else []
        phobert_hist = read_json(phobert_path) if phobert_path else []
        
        if lstm_hist:
            print(f" -> Found LSTM histories for V{version} at: {lstm_path}")
            plot_history(lstm_hist, f"LSTM Training History", "lstm", version)
        else:
            print(f" -> [!] LSTM histories not found for V{version}")
            
        if phobert_hist:
            print(f" -> Found PhoBERT histories for V{version} at: {phobert_path}")
            plot_history(phobert_hist, f"PhoBERT Training History", "phobert", version)
        else:
            print(f" -> [!] PhoBERT histories not found for V{version}")

        # 2. Tìm và đọc files comparison CSV
        lstm_comp = get_comparison_df("lstm", version, lstm_hist)
        phobert_comp = get_comparison_df("phobert", version, phobert_hist)
        
        # 3. Vẽ biểu đồ so sánh đơn lẻ từng phiên bản & lấy mô hình tốt nhất
        best_lstm, best_phobert = plot_single_version_comparison(lstm_comp, phobert_comp, version)
        
        # Lưu trữ kết quả tốt nhất để vẽ Master chart
        if best_lstm is not None:
            best_results_list.append({
                'Version': f'V{version}',
                'Model': 'LSTM',
                'Val_F1': float(best_lstm['Val_F1']),
                'Val_Acc': float(best_lstm.get('Val_Acc', 0.0)),
                'Val_Precision': float(best_lstm.get('Val_Precision', 0.0)),
                'Val_Recall': float(best_lstm.get('Val_Recall', 0.0))
            })
            
        if best_phobert is not None:
            best_results_list.append({
                'Version': f'V{version}',
                'Model': 'PhoBERT',
                'Val_F1': float(best_phobert['Val_F1']),
                'Val_Acc': float(best_phobert.get('Val_Acc', 0.0)),
                'Val_Precision': float(best_phobert.get('Val_Precision', 0.0)),
                'Val_Recall': float(best_phobert.get('Val_Recall', 0.0))
            })

    # 4. Vẽ Master Comparison Chart nếu có dữ liệu
    if best_results_list:
        plot_master_comparison(best_results_list)
        # Lưu CSV tổng hợp để tiện theo dõi
        master_df = pd.DataFrame(best_results_list)
        master_csv_path = os.path.join(VISUAL_DIR, "master_version_comparison.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"[+] Saved Master Comparison CSV to {master_csv_path}")
    else:
        print("[!] No best models found from any version to compile master chart.")
        
    print("\n==================================================")
    print("[SUCCESS] COMPLETED PERFORMANCE VISUALIZATION SUCCESSFULLY")
    print("==================================================")



if __name__ == "__main__":
    main()
