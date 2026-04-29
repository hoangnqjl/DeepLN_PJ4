import pandas as pd
import re

def clean_for_match(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=== GLOBAL CLEANING & DEDUPLICATION ===")
    public_train_path = "dataset/public_train.csv"
    
    df = pd.read_csv(public_train_path)
    initial_len = len(df)
    
    # 1. Drop completely empty messages
    df_clean = df.dropna(subset=['post_message', 'label']).copy()
    print(f"Dropped {initial_len - len(df_clean)} empty row(s).")
    
    # 2. Clean text for deduplication
    df_clean['clean_text'] = df_clean['post_message'].apply(clean_for_match)
    
    # 3. Deduplicate
    df_dedup = df_clean.drop_duplicates(subset=['clean_text']).copy()
    final_len = len(df_dedup)
    
    print(f"Dropped {len(df_clean) - final_len} exact duplicate rows.")
    
    # Drop temp column and save
    df_final = df_dedup.drop(columns=['clean_text'])
    df_final.to_csv(public_train_path, index=False, encoding='utf-8')
    print(f"Dataset successfully cleaned and saved. Final size: {final_len} rows.")

if __name__ == "__main__":
    main()
