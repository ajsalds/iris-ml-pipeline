import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def main():
    df = pd.read_csv("data/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, None]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy")

    with mlflow.start_run(run_name="rf-hyperparam-tuning") as run:
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_accuracy", best_score)

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, "IrisModel")

        print(f"Model registered as: IrisModel (version {result.version})")
        print(f"Best params: {best_params}")
        print(f"Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()
