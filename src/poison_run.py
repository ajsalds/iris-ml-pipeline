import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import seaborn as sns


def flip_labels(y, poison_level):
    """
    Label flipping poisoning attack.
    poison_level = percentage of samples whose labels will be flipped.
    """
    y_poisoned = y.copy()
    n_samples = len(y)
    n_poison = int(n_samples * poison_level)

    # random indices to poison
    indices_to_flip = np.random.choice(n_samples, n_poison, replace=False)

    for idx in indices_to_flip:
        original_label = y_poisoned[idx]
        possible_flips = [l for l in np.unique(y) if l != original_label]
        y_poisoned[idx] = np.random.choice(possible_flips)

    return y_poisoned


def log_confusion_matrix(cm, labels, run_name):
    """Save confusion matrix to file and log to MLflow."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {run_name}")
    img_path = f"conf_matrix_{run_name}.png"
    plt.savefig(img_path)
    plt.close()
    mlflow.log_artifact(img_path)


def train_with_poisoning(poison_level):
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_train = flip_labels(y_train, poison_level)

    # MLflow logging
    run_name = f"poison_{int(poison_level*100)}pct"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("model", "LogisticRegression")

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        log_confusion_matrix(cm, iris.target_names, run_name)

        # Log model
        mlflow.sklearn.log_model(model, "model")

    return acc, cm


if __name__ == "__main__":
    mlflow.set_experiment("IRIS_Label__Poisoning")

    poison_levels = [0.05, 0.10, 0.50]

    for p in poison_levels:
        acc, cm = train_with_poisoning(p)
        print("\n==============================")
        print(f" Poisoning Level: {int(p*100)}%")
        print(f" Accuracy: {acc:.4f}")
        print(" Confusion Matrix:")
        print(cm)
        print("==============================")

