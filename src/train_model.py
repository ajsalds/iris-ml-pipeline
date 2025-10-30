#!/usr/bin/env python
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(data_path)

    # Split features and target
    X = df.drop('species', axis=1)
    y = df['species']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save with joblib
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to CSV data")
    parser.add_argument("out", help="Path to save model (.joblib)")
    args = parser.parse_args()

    train_model(args.data, args.out)

if __name__ == "__main__":
    main()
