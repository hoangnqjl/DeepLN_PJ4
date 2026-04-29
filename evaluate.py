import pandas as pd
import json
import os

def main():
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

    VISUAL_DIR = os.path.join(BASE_PATH, "visual")

    # Load comparison data
    try:
        lstm_comp = pd.read_csv(os.path.join(VISUAL_DIR, 'lstm_comparison.csv'))
    except:
        lstm_comp = pd.DataFrame()
        
    try:
        phobert_comp = pd.read_csv(os.path.join(VISUAL_DIR, 'phobert_comparison.csv'))
    except:
        phobert_comp = pd.DataFrame()

    print("# Final Evaluation Report - Fake News Detection")
    print("\n## LSTM Model Experiments")
    if not lstm_comp.empty:
        print(lstm_comp.sort_values(by='Val_F1', ascending=False).to_markdown(index=False))
    else:
        print("No LSTM data found.")

    print("\n## PhoBERT Model Experiments")
    if not phobert_comp.empty:
        print(phobert_comp.sort_values(by='Val_F1', ascending=False).to_markdown(index=False))
    else:
        print("No PhoBERT data found.")

    best_lstm = lstm_comp.iloc[lstm_comp['Val_F1'].idxmax()] if not lstm_comp.empty else None
    best_phobert = phobert_comp.iloc[phobert_comp['Val_F1'].idxmax()] if not phobert_comp.empty else None

    print("\n## Summary Comparison")
    summary = []
    if best_lstm is not None:
        summary.append({
            'Model': 'LSTM', 
            'Best Params': f"DR={best_lstm['Dropout']}, BS={int(best_lstm['BatchSize'])}", 
            'Acc': best_lstm.get('Val_Acc', '-'),
            'Prec': best_lstm.get('Val_Precision', '-'),
            'Rec': best_lstm.get('Val_Recall', '-'),
            'F1': best_lstm['Val_F1']
        })
    if best_phobert is not None:
        summary.append({
            'Model': 'PhoBERT', 
            'Best Params': f"DR={best_phobert['Dropout']}, LR={best_phobert['LearningRate']}", 
            'Acc': best_phobert.get('Val_Acc', '-'),
            'Prec': best_phobert.get('Val_Precision', '-'),
            'Rec': best_phobert.get('Val_Recall', '-'),
            'F1': best_phobert['Val_F1']
        })
    
    if summary:
        print(pd.DataFrame(summary).to_markdown(index=False))
    
    print("\n## Conclusion")
    if best_phobert is not None and best_lstm is not None:
        diff = best_phobert['Val_F1'] - best_lstm['Val_F1']
        if diff > 0:
            print(f"PhoBERT outperformed LSTM by {diff:.4f} in F1-score.")
        else:
            print(f"LSTM outperformed PhoBERT by {-diff:.4f} in F1-score.")

    
if __name__ == "__main__":
    main()
