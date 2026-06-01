# src/train_models.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── Import RBNN and GRNN from separate file ──
from src.models_definition import RBNN, GRNN

os.makedirs('models',        exist_ok=True)
os.makedirs('results/plots', exist_ok=True)



#  FFNN

def build_ffnn(input_dim=2, output_dim=4):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(output_dim,
                     activation='linear')
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
    print(f"Train: {len(X_tr)}  "
          f"Val: {len(X_val)}")
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
                    f"Train: {logs['loss']:.6f}"
                    f" | Val: "
                    f"{logs['val_loss']:.6f}"
                )
                if (epoch + 1) % 100 == 0
                else None
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

    model.save('models/ffnn_model.keras')
    print("   Saved: models/ffnn_model.keras")
    return model


def train_rbnn(X_tr, y_tr):
    print("\n--- Training RBNN ---")
    print("  Centers: 30  |  Spread: 0.8")
    model = RBNN(n_centers=30, spread=0.8)
    model.fit(X_tr, y_tr)
    joblib.dump(model, 'models/rbnn_model.pkl')
    print("✅ RBNN Done!")
    print("   Saved: models/rbnn_model.pkl")
    return model


def train_grnn(X_tr, y_tr):
    print("\n--- Training GRNN ---")
    model = GRNN(sigma=0.4)
    model.fit(X_tr, y_tr)
    joblib.dump(model, 'models/grnn_model.pkl')
    print("✅ GRNN Done!")
    print("   Saved: models/grnn_model.pkl")
    return model