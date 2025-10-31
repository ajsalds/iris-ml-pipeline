from src.inference import predict_species

def test_inference_format():
    sample = [{'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}]
    preds = predict_species("models/iris_model.pkl", sample)
    assert isinstance(preds, list)
    assert len(preds) == 1
