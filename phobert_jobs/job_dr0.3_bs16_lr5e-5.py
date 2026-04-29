import os
import pandas as pd
from phobert_model import run_phobert_experiment, load_valid_experiment_result, lr_to_tag, RESULTS_DIR, BASE_PATH

if __name__ == "__main__":
    # Cố định tham số cho file này
    lr = 5e-05
    dr = 0.3
    bs = 16
    
    run_name = f"phobert_dr{dr}_bs{bs}_lr{lr_to_tag(lr)}"
    print(f"\n=======================================================")
    print(f"KÍCH HOẠT JOB: {run_name}")
    print(f"=======================================================")

    # Đường dẫn dữ liệu
    train_path = os.path.join(BASE_PATH, "processed_data/train.csv")
    val_path = os.path.join(BASE_PATH, "processed_data/val.csv")
    if not os.path.exists(train_path):
        train_path = "processed_data/train.csv"
        val_path = "processed_data/val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    exp_res_path = os.path.join(RESULTS_DIR, run_name, "experiment_results.json")
    
    # Cơ chế kiểm tra bỏ qua
    if os.path.exists(exp_res_path):
        print(f"\n>>> Tổ hợp {run_name} ĐÃ ĐƯỢC TRAIN TRƯỚC ĐÓ.")
        try:
            res = load_valid_experiment_result(exp_res_path, run_name, dr, bs, lr)
            print(f"-> Đã nạp thành công kết quả cũ. Val F1: {res['val_f1']}")
        except Exception as e:
            print(f"[!] File kết quả cũ bị lỗi: {e}. Tiến hành train lại...")
            res = run_phobert_experiment(
                dr, bs, lr,
                train_df['tokenized_message'], train_df['label'],
                val_df['tokenized_message'], val_df['label']
            )
    else:
        print(f"\n>>> Bắt đầu huấn luyện mới cho tổ hợp: {run_name}")
        res = run_phobert_experiment(
            dr, bs, lr,
            train_df['tokenized_message'], train_df['label'],
            val_df['tokenized_message'], val_df['label']
        )
        print(f"✅ Hoàn thành Job {run_name}!")
