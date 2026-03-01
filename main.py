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
    "downloading": False,
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

def _load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download raw MNIST IDX binary files directly.

    Why not fetch_openml / liac-arff?
    The ARFF parser materialises every pixel value as a Python float object
    (~56 bytes overhead each) before converting to numpy.  For 70k × 784
    values that is ~3 GB of intermediate Python objects — guaranteed OOM
    on Render's free 512 MB tier.

    The raw IDX binary files are parsed straight into numpy uint8 arrays
    with no Python-object overhead.  Peak memory during load is ~275 MB,
    well within the 512 MB limit.
    """
    import gzip
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # PyTorch S3 mirror — AWS-hosted, reliable from Render which also runs on AWS.
    # Falls back to Google Cloud Storage if S3 times out.
    MIRRORS = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]
    FILES = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img":  "t10k-images-idx3-ubyte.gz",
        "test_lbl":  "t10k-labels-idx1-ubyte.gz",
    }

    def fetch(name: str) -> bytes:
        filename = FILES[name]
        for base in MIRRORS:
            url = base + filename
            print(f"[training] Downloading {filename} from {base}…")
            try:
                with urllib.request.urlopen(url, context=ctx, timeout=90) as r:
                    return gzip.decompress(r.read())
            except Exception as e:
                print(f"[training] Mirror failed ({e}), trying next…")
        raise RuntimeError(f"All mirrors failed for {filename}")

    # memoryview slicing is zero-copy (unlike bytes[n:] which makes a 47 MB copy).
    # Pre-allocating float32 and filling in-place avoids a second 188 MB
    # intermediate array that .astype() + / 255.0 would otherwise create.
    # Peak memory this way: 47 MB (raw bytes) + 188 MB (X_train) = ~235 MB,
    # vs ~470 MB with the naive approach.

    raw = fetch("train_img")                                    # 47 MB bytes
    mv = memoryview(raw)                                        # zero-copy view
    X_train = np.empty((60000, 784), dtype=np.float32)         # 188 MB, pre-alloc
    X_train[:] = np.frombuffer(mv[16:], dtype=np.uint8).reshape(60000, 784)
    X_train /= 255.0                                            # in-place, no copy
    del raw, mv                                                 # free 47 MB

    raw = fetch("train_lbl")
    y_train = np.frombuffer(memoryview(raw)[8:], dtype=np.uint8).astype(np.int32)
    del raw

    raw = fetch("test_img")                                     # 8 MB bytes
    mv = memoryview(raw)
    X_test = np.empty((10000, 784), dtype=np.float32)          # 31 MB, pre-alloc
    X_test[:] = np.frombuffer(mv[16:], dtype=np.uint8).reshape(10000, 784)
    X_test /= 255.0
    del raw, mv

    raw = fetch("test_lbl")
    y_test = np.frombuffer(memoryview(raw)[8:], dtype=np.uint8).astype(np.int32)
    del raw

    return X_train, y_train, X_test, y_test


def _train_model() -> None:
    import gc
    global model
    app_state["training"] = True
    app_state["downloading"] = True

    try:
        X_train, y_train, X_test, y_test = _load_mnist()
        app_state["downloading"] = False

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

        # Free training data — model weights are ~100 KB; X_train is ~220 MB.
        del X_train, y_train, X_test, y_test
        gc.collect()
        print("[training] Training data freed from memory.")

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
        "downloading": app_state["downloading"],
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
