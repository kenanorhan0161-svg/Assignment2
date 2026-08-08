"""
Inference-only helper for the Stage 1 sentiment checkpoint.

Usage:
    python predict.py public_test.csv predictions.csv
or, after Stage 2 release:
    python predict.py hidden_test.csv hidden_test_predictions.csv
"""
from pathlib import Path
import sys
import joblib
import pandas as pd

CHECKPOINT_PATH = Path("model_checkpoint") / "sentiment_ensemble.joblib"

def load_checkpoint(path=CHECKPOINT_PATH):
    return joblib.load(path)

def predict_texts(texts, checkpoint):
    texts = pd.Series(texts).fillna("").astype(str)
    word_scores = checkpoint["word_model"].decision_function(texts)
    char_scores = checkpoint["char_model"].decision_function(texts)
    scores = (
        checkpoint["alpha"] * word_scores
        + (1.0 - checkpoint["alpha"]) * char_scores
    )
    return (scores >= checkpoint["threshold"]).astype(int)

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    checkpoint = load_checkpoint()
    pred = predict_texts(df["text"], checkpoint)
    pd.DataFrame({
        "id": df["id"],
        "predicted_label": pred.astype(int),
    }).to_csv(output_csv, index=False)
    print(f"Saved {len(pred)} predictions to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python predict.py INPUT.csv OUTPUT.csv")
    main(sys.argv[1], sys.argv[2])
