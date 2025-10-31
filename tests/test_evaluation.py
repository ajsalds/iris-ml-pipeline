from src.evaluate_model import evaluate_model

def test_evaluate_runs():
    acc = evaluate_model("models/iris_model.pkl", "data/iris.csv")
    assert acc >= 0.5
