import os

lrs = [2e-5, 3e-5, 5e-5]
dropouts = [0.1, 0.3, 0.5]
batch_sizes = [8, 16, 32]

# Tạo thư mục chứa 27 job riêng biệt
output_folder = "phobert_jobs"
os.makedirs(output_folder, exist_ok=True)

for lr in lrs:
    for dr in dropouts:
        for bs in batch_sizes:
            lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")
            file_name = os.path.join(output_folder, f"job_dr{dr}_bs{bs}_lr{lr_tag}.py")
            
            # Nội dung code cho từng file độc lập
            code_content = f"""import os
import pandas as pd
from phobert_model import run_phobert_experiment, load_valid_experiment_result, lr_to_tag, RESULTS_DIR, BASE_PATH

if __name__ == "__main__":
    # Cố định tham số cho file này
    lr = {lr}
    dr = {dr}
    bs = {bs}
    
    run_name = f"phobert_dr{{dr}}_bs{{bs}}_lr{{lr_to_tag(lr)}}"
    print(f"\\n=======================================================")
    print(f"KÍCH HOẠT JOB: {{run_name}}")
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
        print(f"\\n>>> Tổ hợp {{run_name}} ĐÃ ĐƯỢC TRAIN TRƯỚC ĐÓ.")
        try:
            res = load_valid_experiment_result(exp_res_path, run_name, dr, bs, lr)
            print(f"-> Đã nạp thành công kết quả cũ. Val F1: {{res['val_f1']}}")
        except Exception as e:
            print(f"[!] File kết quả cũ bị lỗi: {{e}}. Tiến hành train lại...")
            res = run_phobert_experiment(
                dr, bs, lr,
                train_df['tokenized_message'], train_df['label'],
                val_df['tokenized_message'], val_df['label']
            )
    else:
        print(f"\\n>>> Bắt đầu huấn luyện mới cho tổ hợp: {{run_name}}")
        res = run_phobert_experiment(
            dr, bs, lr,
            train_df['tokenized_message'], train_df['label'],
            val_df['tokenized_message'], val_df['label']
        )
        print(f"✅ Hoàn thành Job {{run_name}}!")
"""
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(code_content)

print(f"🎉 Đã tạo thành công 27 file Python độc lập trong thư mục '{output_folder}'!")
