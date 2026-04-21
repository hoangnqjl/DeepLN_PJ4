import pandas as pd
import json

def main():
    # Load comparison data
    try:
        lstm_comp = pd.read_csv('results/lstm_comparison.csv')
    except:
        lstm_comp = pd.DataFrame()
        
    try:
        phobert_comp = pd.read_csv('results/phobert_comparison.csv')
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
        summary.append({'Model': 'LSTM', 'Best Params': f"DR={best_lstm['Dropout']}, BS={int(best_lstm['BatchSize'])}", 'Best F1': best_lstm['Val_F1']})
    if best_phobert is not None:
        summary.append({'Model': 'PhoBERT', 'Best Params': f"DR={best_phobert['Dropout']}, LR={best_phobert['LearningRate']}", 'Best F1': best_phobert['Val_F1']})
    
    if summary:
        print(pd.DataFrame(summary).to_markdown(index=False))
    
    print("\n## Conclusion")
    if best_phobert is not None and best_lstm is not None:
        diff = best_phobert['Best F1'] - best_lstm['Best F1']
        if diff > 0:
            print(f"PhoBERT outperformed LSTM by {diff:.4f} in F1-score.")
        else:
            print(f"LSTM outperformed PhoBERT by {-diff:.4f} in F1-score.")
    
if __name__ == "__main__":
    main()
