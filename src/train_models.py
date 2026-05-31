# src/train_models.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

os.makedirs('models',        exist_ok=True)
os.makedirs('results/plots', exist_ok=True)


# ══════════════════════════════
#  FFNN
# ══════════════════════════════
def build_ffnn(input_dim=2, output_dim=4):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_ffnn(X_tr, y_tr, X_val, y_val,
               epochs=3000):
    print("\n--- Training FFNN ---")
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}")
    print(f"Max epochs: {epochs}")
    print("-" * 55)

    model = build_ffnn()
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=200,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=50,
            min_lr=1e-8,
            verbose=1
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs:
                print(
                    f"  Ep {epoch+1:4d} | "
                    f"Train: {logs['loss']:.6f} | "
                    f"Val: {logs['val_loss']:.6f}"
                )
                if (epoch + 1) % 100 == 0 else None
        )
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        callbacks=cb,
        verbose=0
    )

    final = history.history['val_loss'][-1]
    print(f"\n✅ FFNN Done!")
    print(f"   Val Loss : {final:.6f}")
    print(f"   Epochs   : "
          f"{len(history.history['loss'])}")

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],
             label='Train Loss')
    plt.plot(history.history['val_loss'],
             label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('FFNN Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        'results/plots/ffnn_training_history.png',
        dpi=150
    )
    plt.close()
    print("   Loss plot saved.")
    model.save('models/ffnn_model.keras')
    print("   Saved: models/ffnn_model.keras")
    return model


# ══════════════════════════════
#  RBNN
# ══════════════════════════════
class RBNN:
    def __init__(self, n_centers=30, spread=0.8):
        self.n_centers = n_centers
        self.spread    = spread

    def _rbf(self, X, centers):
        diff = (X[:, np.newaxis, :]
                - centers[np.newaxis, :, :])
        dist = np.sum(diff ** 2, axis=2)
        return np.exp(
            -dist / (2 * self.spread ** 2)
        )

    def fit(self, X, y):
        km = KMeans(
            n_clusters=self.n_centers,
            random_state=42,
            n_init=20
        )
        km.fit(X)
        self.centers = km.cluster_centers_
        Phi = self._rbf(X, self.centers)
        self.out = Ridge(alpha=0.001)
        self.out.fit(Phi, y)
        mse = mean_squared_error(
            y, self.out.predict(Phi)
        )
        print(f"  RBNN Train MSE : {mse:.6f}")
        return self

    def predict(self, X):
        return self.out.predict(
            self._rbf(X, self.centers)
        )


def train_rbnn(X_tr, y_tr):
    print("\n--- Training RBNN ---")
    print("  Centers: 30  |  Spread: 0.8")
    model = RBNN(n_centers=30, spread=0.8)
    model.fit(X_tr, y_tr)
    joblib.dump(model, 'models/rbnn_model.pkl')
    print("✅ RBNN Done!")
    print("   Saved: models/rbnn_model.pkl")
    return model


# ══════════════════════════════
#  GRNN
# ══════════════════════════════
class GRNN:
    def __init__(self, sigma=0.4):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_tr = X.copy()
        self.y_tr = y.copy()
        print(f"  GRNN samples : {len(X)}  "
              f"sigma : {self.sigma}")
        return self

    def predict(self, X):
        preds = np.zeros(
            (len(X), self.y_tr.shape[1])
        )
        for i, x in enumerate(X):
            diff  = self.X_tr - x
            dist2 = np.sum(diff ** 2, axis=1)
            w     = np.exp(
                -dist2 / (2 * self.sigma ** 2)
            )
            w_sum = w.sum()
            if w_sum < 1e-300:
                preds[i] = self.y_tr[
                    np.argmin(dist2)
                ]
            else:
                preds[i] = (
                    w @ self.y_tr
                ) / w_sum
        return preds


def train_grnn(X_tr, y_tr):
    print("\n--- Training GRNN ---")
    model = GRNN(sigma=0.4)
    model.fit(X_tr, y_tr)
    joblib.dump(model, 'models/grnn_model.pkl')
    print("✅ GRNN Done!")
    print("   Saved: models/grnn_model.pkl")
    return model