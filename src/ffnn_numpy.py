# src/ffnn_numpy.py
import numpy as np

class FFNNNumpy:
    def __init__(self, weights_path):
        self.weights = np.load(
            weights_path,
            allow_pickle=True
        )
        print(f"FFNNNumpy loaded: "
              f"{len(self.weights)} arrays")

    def _relu(self, x):
        return np.maximum(0, x)

    def predict(self, X):
        out      = np.array(X, dtype=float)
        n_layers = len(self.weights) // 2
        for i in range(n_layers):
            W   = self.weights[i * 2]
            b   = self.weights[i * 2 + 1]
            out = out @ W + b
            if i < n_layers - 1:
                out = self._relu(out)
        return out