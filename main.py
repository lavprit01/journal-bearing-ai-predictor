# main.py

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from data.table86_data    import *
from src.data_prep        import load_data, explore_data, prepare_data
from src.train_models     import train_ffnn, train_rbnn, train_grnn


from src.evaluate         import (evaluate_model,
                                  plot_predictions,
                                  regression_plot,
                                  plot_error_comparison,
                                  time_model,
                                  PARAM_NAMES) 

os.makedirs('models',        exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

print("\n" + "="*50)
print("  JOURNAL BEARING AI PROJECT")
print("  3 Models: FFNN | RBNN | GRNN")
print("="*50)

#  Step 1: Load data 
print("\n[1/6] Loading data...")
df = load_data('data/table86_data.csv')
explore_data(df)

#  Step 2: Prepare data 
print("\n[2/6] Preparing data...")
(X_tr, X_val, X_te,
 y_tr, y_val, y_te,
 scaler_X, scaler_y) = prepare_data(df)

joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
print("     Scalers saved.")

#  Step 3: Train FFNN 
print("\n[3/6] Training FFNN...")
ffnn = train_ffnn(X_tr, y_tr, X_val, y_val,
                  epochs=3000)

#  Step 4: Train RBNN 
print("\n[4/6] Training RBNN...")
rbnn = train_rbnn(X_tr, y_tr)

#  Step 5: Train GRNN 
print("\n[5/6] Training GRNN...")
grnn = train_grnn(X_tr, y_tr)

#  Step 6: Evaluate 
print("\n[6/6] Evaluating all models...")

pred_ffnn = ffnn.predict(X_te, verbose=0)
pred_rbnn = rbnn.predict(X_te)
pred_grnn = grnn.predict(X_te)

m_ffnn = evaluate_model('FFNN', y_te, pred_ffnn)
m_rbnn = evaluate_model('RBNN', y_te, pred_rbnn)
m_grnn = evaluate_model('GRNN', y_te, pred_grnn)

#  Error comparison table 

print("\n" + "="*85)
print("  FINAL ERROR COMPARISON TABLE (on test set)")
print("="*85)
print(f"{'Parameter':<25} {'FFNN MAPE%':>12} "
      f"{'RBNN MAPE%':>12} {'GRNN MAPE%':>12}  Best")
print("-" * 85)

for p in PARAM_NAMES:
    fm = float(m_ffnn.loc[p, 'MAPE (%)'])
    rm = float(m_rbnn.loc[p, 'MAPE (%)'])
    gm = float(m_grnn.loc[p, 'MAPE (%)'])
    best_val = min(fm, rm, gm)
    best_name = (
        'FFNN' if best_val == fm else
        'RBNN' if best_val == rm else 'GRNN'
    )
    print(f"{p:<25} {fm:>11.2f}% "
          f"{rm:>11.2f}% {gm:>11.2f}%  "
          f"→ {best_name}")
print("="*85)

#  Plots 
print("\nSaving plots...")
regression_plot(y_te, pred_ffnn, 'FFNN')
regression_plot(y_te, pred_rbnn, 'RBNN')
regression_plot(y_te, pred_grnn, 'GRNN')

plot_predictions(
    y_te,
    {'FFNN': pred_ffnn,
     'RBNN': pred_rbnn,
     'GRNN': pred_grnn},
    title_suffix='test_set'
)

plot_error_comparison(m_ffnn, m_rbnn, m_grnn)

#  Timing 
print("\n--- Prediction Times ---")
t_ffnn = time_model(ffnn, X_te, 'keras')
t_rbnn = time_model(rbnn, X_te, 'custom')
t_grnn = time_model(grnn, X_te, 'custom')
print(f"  FFNN : {t_ffnn:.4f} s")
print(f"  RBNN : {t_rbnn:.6f} s")
print(f"  GRNN : {t_grnn:.6f} s")

print("\n" + "="*50)
print("✅ ALL DONE!")
print("   Plots → results/plots/")
print("   Models → models/")
print("="*50)