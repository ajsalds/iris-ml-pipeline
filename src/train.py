import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    df = pd.read_csv("data/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    rf = RandomForestClassifier(random_state=42)

    with mlflow.start_run(run_name="rf-training") as run:
        rf.fit(X, y)

        train_preds = rf.predict(X)
        train_accuracy = accuracy_score(y, train_preds)

        # Log model parameters and training accuracy
        mlflow.log_params(rf.get_params())
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Log model to MLflow
        mlflow.sklearn.log_model(rf, "model")

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Model logged in run: {run.info.run_id}")

if __name__ == "__main__":
    main()
