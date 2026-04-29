import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_for_match(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    public_train_path = "dataset/public_train.csv"
    new_data_path = "new_data/train_data.csv"
    
    df_public = pd.read_csv(public_train_path)
    df_new = pd.read_csv(new_data_path)
    
    print(f"Total rows in public_train.csv: {len(df_public)}")
    print(f"Total rows in train_data.csv: {len(df_new)}")
    
    df_new_fake = df_new[df_new['label'] == 1].copy()
    print(f"Fake rows in train_data.csv: {len(df_new_fake)}")
    
    df_public = df_public.dropna(subset=['post_message'])
    df_new_fake = df_new_fake.dropna(subset=['content'])
    
    print("\n[Round 1] Substring Matching with public_train...")
    df_public['clean_text_sub'] = df_public['post_message'].apply(lambda x: re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]', '', x.lower()))
    df_new_fake['clean_text_sub'] = df_new_fake['content'].apply(lambda x: re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]', '', x.lower()))
    
    public_texts_sub = [t for t in df_public['clean_text_sub'].tolist() if len(t) > 20]
    
    sub_duplicates = []
    for idx, row in df_new_fake.iterrows():
        new_text = row['clean_text_sub']
        if len(new_text) <= 20:
            continue
        for pub_text in public_texts_sub:
            if new_text in pub_text or pub_text in new_text:
                sub_duplicates.append(idx)
                break
                
    print(f"-> Found {len(sub_duplicates)} duplicates via Substring.")
    
    print("\n[Round 2] TF-IDF Cosine Similarity with public_train...")
    df_public['clean_tfidf'] = df_public['post_message'].apply(clean_for_match)
    df_new_fake['clean_tfidf'] = df_new_fake['content'].apply(clean_for_match)
    
    vectorizer = TfidfVectorizer()
    all_corpus = df_public['clean_tfidf'].tolist() + df_new_fake['clean_tfidf'].tolist()
    vectorizer.fit(all_corpus)
    
    tfidf_public = vectorizer.transform(df_public['clean_tfidf'])
    tfidf_new = vectorizer.transform(df_new_fake['clean_tfidf'])
    
    sim_matrix = cosine_similarity(tfidf_new, tfidf_public)
    max_sim_values = np.max(sim_matrix, axis=1)
    
    similarity_duplicates = []
    SIM_THRESHOLD = 0.80
    
    for i in range(len(df_new_fake)):
        idx = df_new_fake.index[i]
        if max_sim_values[i] >= SIM_THRESHOLD:
            if idx not in sub_duplicates:
                similarity_duplicates.append(idx)
                
    print(f"-> Found additional {len(similarity_duplicates)} duplicates via TF-IDF.")
    
    all_dup_indices = list(set(sub_duplicates + similarity_duplicates))
    df_uniques = df_new_fake.loc[~df_new_fake.index.isin(all_dup_indices)].copy()
    
    print("\n[Round 3] Checking for INTERNAL duplicates in new unique data...")
    # Clean internal duplicates
    df_uniques['clean_internal'] = df_uniques['content'].apply(clean_for_match)
    
    # Simple drop_duplicates on clean_internal
    before_internal = len(df_uniques)
    df_uniques_final = df_uniques.drop_duplicates(subset=['clean_internal']).copy()
    after_internal = len(df_uniques_final)
    internal_dups = before_internal - after_internal
    print(f"-> Found {internal_dups} INTERNAL duplicates.")
    
    print(f"\nFinal Deduplication Summary:")
    print(f"- Duplicates against public_train: {len(all_dup_indices)}")
    print(f"- Duplicates within new data: {internal_dups}")
    print(f"- Total new unique fake news: {after_internal}")
    
    output_path = "new_data/new_unique_fake.csv"
    df_uniques_final.drop(columns=['clean_text_sub', 'clean_tfidf', 'clean_internal']).to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {after_internal} strictly unique fake news records to {output_path}")

if __name__ == "__main__":
    main()
