# main.py
import os
import sys
import joblib
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_prep import load_data, explore_data, prepare_data
from src.train_models import train_ffnn, train_rbnn, train_grnn
from src.evaluate import evaluate_model, plot_predictions, regression_plot, plot_error_comparison, time_model, PARAM_NAMES

print("\n" + "="*50)
print("  JOURNAL BEARING AI PROJECT (8 Parameters)")
print("="*50)

df = load_data('data/table86_data.csv')
explore_data(df)

X_tr, X_val, X_te, y_tr, y_val, y_te, scaler_X, scaler_y = prepare_data(df)

joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

print("\nTraining FFNN...")
ffnn = train_ffnn(X_tr, y_tr, X_val, y_val, epochs=3000)
print("\nTraining RBNN...")
rbnn = train_rbnn(X_tr, y_tr)
print("\nTraining GRNN...")
grnn = train_grnn(X_tr, y_tr)

pred_ffnn = ffnn.predict(X_te, verbose=0)
pred_rbnn = rbnn.predict(X_te)
pred_grnn = grnn.predict(X_te)

m_ffnn = evaluate_model('FFNN', y_te, pred_ffnn)
m_rbnn = evaluate_model('RBNN', y_te, pred_rbnn)
m_grnn = evaluate_model('GRNN', y_te, pred_grnn)

print("\n" + "="*85)
print(f"{'Parameter':<25} {'FFNN MAPE%':>12} {'RBNN MAPE%':>12} {'GRNN MAPE%':>12}  Best")
print("-" * 85)
for p in PARAM_NAMES:
    fm, rm, gm = float(m_ffnn.loc[p, 'MAPE (%)']), float(m_rbnn.loc[p, 'MAPE (%)']), float(m_grnn.loc[p, 'MAPE (%)'])
    best = 'FFNN' if min(fm,rm,gm) == fm else ('RBNN' if min(fm,rm,gm) == rm else 'GRNN')
    print(f"{p:<25} {fm:>11.2f}% {rm:>11.2f}% {gm:>11.2f}%  → {best}")

print("\nSaving plots...")
regression_plot(y_te, pred_ffnn, 'FFNN')
regression_plot(y_te, pred_rbnn, 'RBNN')
regression_plot(y_te, pred_grnn, 'GRNN')
plot_predictions(y_te, {'FFNN': pred_ffnn, 'RBNN': pred_rbnn, 'GRNN': pred_grnn}, 'test_set')
plot_error_comparison(m_ffnn, m_rbnn, m_grnn)

print("\n✅ ALL DONE! Check models/ and results/plots/")