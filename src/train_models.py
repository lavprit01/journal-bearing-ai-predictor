# src/train_models.py
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from src.models_definition import RBNN, GRNN

os.makedirs('models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

def build_ffnn(input_dim=2, output_dim=8):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_ffnn(X_tr, y_tr, X_val, y_val, epochs=3000):
    model = build_ffnn()
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True, verbose=1),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-8, verbose=1)]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, callbacks=cb, verbose=0)
    model.save('models/ffnn_model.keras')
    return model

def train_rbnn(X_tr, y_tr):
    model = RBNN(n_centers=30, spread=0.8).fit(X_tr, y_tr)
    joblib.dump(model, 'models/rbnn_model.pkl')
    return model

def train_grnn(X_tr, y_tr):
    model = GRNN(sigma=0.4).fit(X_tr, y_tr)
    joblib.dump(model, 'models/grnn_model.pkl')
    return model