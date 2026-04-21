import pandas as pd
import numpy as np
import os
import re
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers (optional, keeping for now)
    # text = re.sub(r'[^\w\s]', '', text)
    # text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Drop rows with missing post_message or label
    df = df.dropna(subset=['post_message', 'label'])
    
    print("Cleaning text and tokenizing...")
    # Clean text
    df['clean_message'] = df['post_message'].apply(clean_text)
    
    # Vietnamese word segmentation
    df['tokenized_message'] = df['clean_message'].apply(lambda x: ViTokenizer.tokenize(x))
    
    # No sample print to avoid console encoding issues on Windows
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"\nData split complete:")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Label distribution in Train:\n{train_df['label'].value_counts(normalize=True)}")
    
    return train_df, val_df

if __name__ == "__main__":
    dataset_path = "dataset/public_train.csv"
    train_df, val_df = preprocess_data(dataset_path)
    
    # Save processed data
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    print(f"\nProcessed data saved to {output_dir}")
