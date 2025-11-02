from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.joblib")

app = FastAPI(title="Iris API")

class IrisSample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to Iris Classifier API!"}

@app.post("/predict")
def predict(data: IrisSample):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    print(pred)
    return {
            "prediction": pred
            }
