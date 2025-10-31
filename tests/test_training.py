import os
from src.train_model import train_model

def test_training_creates_model(tmp_path):
    out = tmp_path / "model.pkl"
    model = train_model("data/iris.csv", str(out))
    assert os.path.exists(out)
