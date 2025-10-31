import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_latest_registered(model_name="IrisModel"):
    client = mlflow.tracking.MlflowClient()
    # Get the latest registered version
    versions = client.get_latest_versions(model_name)
    latest_version = versions[-1].version  # pick the most recent one

    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv("data/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, preds))
    return acc

if __name__ == "__main__":
    evaluate_latest_registered()
