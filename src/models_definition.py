# src/models_definition.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class RBNN:
    def __init__(self, n_centers=30, spread=0.8):
        self.n_centers = n_centers
        self.spread    = spread

    def _rbf(self, X, centers):
        diff = (X[:, np.newaxis, :] - centers[np.newaxis, :, :])
        dist = np.sum(diff ** 2, axis=2)
        return np.exp(-dist / (2 * self.spread ** 2))

    def fit(self, X, y):
        km = KMeans(n_clusters=self.n_centers, random_state=42, n_init=20)
        km.fit(X)
        self.centers = km.cluster_centers_
        Phi = self._rbf(X, self.centers)
        self.out = Ridge(alpha=0.001)
        self.out.fit(Phi, y)
        mse = mean_squared_error(y, self.out.predict(Phi))
        print(f"  RBNN Train MSE : {mse:.6f}")
        return self

    def predict(self, X):
        return self.out.predict(self._rbf(X, self.centers))

class GRNN:
    def __init__(self, sigma=0.4):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_tr = X.copy()
        self.y_tr = y.copy()
        return self

    def predict(self, X):
        preds = np.zeros((len(X), self.y_tr.shape[1]))
        for i, x in enumerate(X):
            diff  = self.X_tr - x
            dist2 = np.sum(diff ** 2, axis=1)
            w     = np.exp(-dist2 / (2 * self.sigma ** 2))
            w_sum = w.sum()
            if w_sum < 1e-300:
                preds[i] = self.y_tr[np.argmin(dist2)]
            else:
                preds[i] = (w @ self.y_tr) / w_sum
        return preds