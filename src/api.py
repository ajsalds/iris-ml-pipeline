from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time
import json
import logging

# --- OpenTelemetry imports ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter


# ------------------------------
#  Setup OpenTelemetry Tracing
# ------------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)


# ------------------------------
#  Setup Structured Logging
# ------------------------------
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(
    json.dumps({
        "severity": "%(levelname)s",
        "message": "%(message)s",
        "timestamp": "%(asctime)s"
    })
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------
#  FastAPI app initialization
# ------------------------------
app = FastAPI(title="Iris Classifier API")

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.joblib")

app_state = {"is_ready": False, "is_alive": True}
model = None


# ------------------------------
#  Startup event – load model
# ------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        time.sleep(2)  # simulate model load latency
        model = joblib.load(MODEL_PATH)
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "model_loaded", "path": MODEL_PATH}))
    except Exception as e:
        app_state["is_ready"] = False
        logger.exception(json.dumps({"event": "model_load_failed", "error": str(e)}))


# ------------------------------
#  Health / Readiness endpoints
# ------------------------------
@app.get("/live_check", tags=["Probe"])
def live_check():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
def ready_check():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# ------------------------------
#  Request model
# ------------------------------
class IrisSample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ------------------------------
#  Middleware: track latency
# ------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response


# ------------------------------
#  Global Exception Handler
# ------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id}
    )


# ------------------------------
#  Root and Prediction Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Iris Classifier API!"}


@app.post("/predict")
def predict(data: IrisSample):
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    with tracer.start_as_current_span("model_inference") as span:
        start = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            df = pd.DataFrame([data.dict()])
            prediction = model.predict(df)[0]
            latency = round((time.time() - start) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": data.dict(),
                "prediction": prediction,
                "latency_ms": latency,
                "status": "success"
            }))

            return {"prediction": prediction, "trace_id": trace_id}

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
