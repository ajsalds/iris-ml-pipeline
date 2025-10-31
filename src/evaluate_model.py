#!/usr/bin/env python
import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)
    y = df['species']
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"accuracy: {acc:.4f}")
    print(classification_report(y, preds))
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model pickle")
    parser.add_argument("data", help="Path to CSV data")
    args = parser.parse_args()
    evaluate_model(args.model, args.data)

if __name__ == "__main__":
    main()
