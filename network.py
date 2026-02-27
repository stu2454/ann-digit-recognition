"""
Hand-coded Multi-Layer Perceptron using NumPy only.
Architecture: 784 → 32 → 16 → 10
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes: list[int], lr: float = 0.1):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(42)
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation (good for ReLU layers)
            scale = np.sqrt(2.0 / fan_in)
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Subtract row-max for numerical stability
        z_shifted = z - z.max(axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Returns (activations, pre_activations).
        activations[0] = input X
        activations[k] = post-activation of layer k
        pre_activations[k] = z before activation of layer k
        """
        activations = [X]
        pre_activations: list[np.ndarray] = []

        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
            activations.append(a)
            current = a

        return activations, pre_activations

    # ------------------------------------------------------------------
    # Backward pass (cross-entropy + softmax gradient simplification)
    # ------------------------------------------------------------------

    def _backward(
        self,
        y_onehot: np.ndarray,
        activations: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (dW, db) per layer, in forward order.
        Uses the simplified softmax + cross-entropy output gradient.
        """
        n = activations[0].shape[0]
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.weights)  # type: ignore

        # Output layer gradient: softmax + cross-entropy simplifies to (ŷ - y)
        delta = activations[-1] - y_onehot  # (n, 10)

        for i in reversed(range(len(self.weights))):
            dW = activations[i].T @ delta / n
            db = delta.mean(axis=0)
            grads[i] = (dW, db)
            if i > 0:
                # Propagate through ReLU: multiply by d(ReLU)/dz = 1 where activation > 0
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)

        return grads

    def _update_weights(
        self, grads: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for i, (dW, db) in enumerate(grads):
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        progress_callback=None,
    ) -> list[float]:
        """
        Mini-batch SGD with 5% LR decay per epoch.
        progress_callback(epoch, total_epochs, accuracy) called after each epoch.
        Returns per-epoch accuracy list.
        """
        n = len(X_train)
        accuracy_history: list[float] = []
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, n, batch_size):
                Xb = X_shuf[start : start + batch_size]
                yb = y_shuf[start : start + batch_size]
                y_onehot = np.eye(10, dtype=np.float32)[yb]

                activations, _ = self.forward(Xb)
                grads = self._backward(y_onehot, activations)
                self._update_weights(grads)

            # Evaluate on a subsample to keep startup time reasonable
            acc = self.accuracy(X_train[:5000], y_train[:5000])
            accuracy_history.append(acc)

            # 5% LR decay per epoch
            self.lr *= 0.95

            if progress_callback:
                progress_callback(epoch + 1, epochs, acc)

        return accuracy_history

    # ------------------------------------------------------------------
    # Online fine-tuning (used by /feedback endpoint)
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.005,
        steps: int = 8,
    ) -> None:
        """
        Run `steps` gradient updates on a small labelled batch.
        Uses a much smaller LR than training to avoid catastrophic forgetting.
        X: (n, 784)  y: (n,) integer labels
        """
        y_onehot = np.eye(10, dtype=np.float32)[y]
        for _ in range(steps):
            activations, _ = self.forward(X)
            grads = self._backward(y_onehot, activations)
            for i, (dW, db) in enumerate(grads):
                self.weights[i] -= lr * dW
                self.biases[i] -= lr * db

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        activations, _ = self.forward(X)
        preds = np.argmax(activations[-1], axis=1)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------
    # Inference (used by API)
    # ------------------------------------------------------------------

    def predict_with_activations(self, x: np.ndarray) -> dict:
        """
        x: shape (1, 784), values in [0, 1].
        Returns structured dict for the frontend animation.
        """
        activations, _ = self.forward(x)
        probs = activations[-1][0]  # (10,)
        prediction = int(np.argmax(probs))

        return {
            "prediction": prediction,
            "probabilities": probs.tolist(),
            "activations": {
                "hidden1": activations[1][0].tolist(),  # (32,)
                "hidden2": activations[2][0].tolist(),  # (16,)
                "output": activations[3][0].tolist(),   # (10,) post-softmax
            },
        }
