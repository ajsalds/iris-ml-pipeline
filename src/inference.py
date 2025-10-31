import joblib
import pandas as pd

def predict_species(model_path: str, input_data):
    model = joblib.load(model_path)
    df = pd.DataFrame(input_data)
    preds = model.predict(df)
    return preds.tolist()

if __name__ == "__main__":
    # example usage
    import sys, json
    model = sys.argv[1]
    sample = json.loads(sys.argv[2])  # pass JSON list of dicts
    print(predict_species(model, sample))
