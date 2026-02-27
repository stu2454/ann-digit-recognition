"""
FastAPI application.
Trains the MLP on MNIST in a background thread at startup.
The frontend polls /status until training is complete.
"""
import threading
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from network import MLP

# ---------------------------------------------------------------------------
# Global state
# Written by the training thread, read by FastAPI request handlers.
# Simple dict assignments on primitive types are GIL-safe in CPython.
# ---------------------------------------------------------------------------
app_state: dict = {
    "trained": False,
    "training": False,
    "accuracy": 0.0,
    "epoch": 0,
    "total_epochs": 20,
    "error": None,
}
model: MLP | None = None

# ---------------------------------------------------------------------------
# Replay buffer — stores recent feedback examples to include in fine-tune
# updates, reducing catastrophic forgetting of older digits.
# Protected by a lock because feedback requests can arrive concurrently.
# ---------------------------------------------------------------------------
import collections
_feedback_lock = threading.Lock()
_replay_buffer: collections.deque = collections.deque(maxlen=64)  # (x, y) pairs


# ---------------------------------------------------------------------------
# Training (runs in a daemon thread)
# ---------------------------------------------------------------------------

def _train_model() -> None:
    global model
    app_state["training"] = True

    try:
        import ssl
        import sklearn.datasets as _skds
        from sklearn.datasets import fetch_openml

        # macOS ships without the required CA bundle for Python's urllib.
        # This unverified context is fine for fetching a public dataset.
        ssl._create_default_https_context = ssl._create_unverified_context

        print("[training] Fetching MNIST (this may take a moment on first run)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")

        # Float32 to stay within Render free-tier 512MB RAM
        X = mnist.data.astype(np.float32) / 255.0
        # fetch_openml returns labels as strings in recent sklearn versions
        y = mnist.target.astype(np.int32)

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        print("[training] Starting training: 784→32→16→10, 20 epochs...")
        model = MLP(layer_sizes=[784, 32, 16, 10], lr=0.1)

        def _progress(epoch: int, total: int, acc: float) -> None:
            app_state["epoch"] = epoch
            app_state["total_epochs"] = total
            app_state["accuracy"] = round(acc, 4)
            print(f"[training] Epoch {epoch}/{total}  train-acc={acc:.4f}")

        model.train(X_train, y_train, epochs=20, batch_size=64, progress_callback=_progress)

        test_acc = model.accuracy(X_test, y_test)
        app_state["accuracy"] = round(test_acc, 4)
        app_state["trained"] = True
        print(f"[training] Done. Test accuracy: {test_acc:.4f}")

    except Exception as exc:
        app_state["error"] = str(exc)
        print(f"[training] Failed: {exc}")
    finally:
        app_state["training"] = False


# ---------------------------------------------------------------------------
# App lifespan — kick off background training thread on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_train_model, daemon=True, name="training")
    thread.start()
    yield
    # Nothing to clean up on shutdown


app = FastAPI(title="ANN Digit Recognition", lifespan=lifespan)

# Static files (HTML / CSS / JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/status")
async def status():
    return {
        "trained": app_state["trained"],
        "training": app_state["training"],
        "accuracy": app_state["accuracy"],
        "epoch": app_state["epoch"],
        "total_epochs": app_state["total_epochs"],
        "error": app_state["error"],
    }


class PredictRequest(BaseModel):
    pixels: list[float]  # 784 floats in [0, 1]


@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None or not app_state["trained"]:
        return {"error": "Model not ready yet — training in progress"}

    x = np.array(request.pixels, dtype=np.float32).reshape(1, 784)
    result = model.predict_with_activations(x)
    return result


class FeedbackRequest(BaseModel):
    pixels: list[float]        # 784 floats in [0, 1]
    correct_label: int         # 0-9


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    if model is None or not app_state["trained"]:
        return {"error": "Model not ready yet"}
    if not (0 <= request.correct_label <= 9):
        return {"error": "correct_label must be 0-9"}

    x = np.array(request.pixels, dtype=np.float32).reshape(1, 784)
    y = np.array([request.correct_label], dtype=np.int32)

    with _feedback_lock:
        _replay_buffer.append((x, y))

        # Build a batch: this correction + a random sample of past corrections
        all_x = [x]
        all_y = [y]
        past = list(_replay_buffer)[:-1]  # exclude the one just added
        if past:
            import random
            sample = random.sample(past, min(len(past), 7))
            for px, py in sample:
                all_x.append(px)
                all_y.append(py)

        batch_x = np.concatenate(all_x, axis=0)
        batch_y = np.concatenate(all_y, axis=0)
        model.fine_tune(batch_x, batch_y, lr=0.005, steps=8)

    print(f"[feedback] Corrected → {request.correct_label}  (buffer size: {len(_replay_buffer)})")
    return {"status": "ok", "buffer_size": len(_replay_buffer)}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
