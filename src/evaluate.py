# src/evaluate.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score

PARAM_NAMES = ['Sommerfeld S', 'Leakage Flow QL', 'Inlet Flow Qi',
               'Friction f(R/C)', 'Max Pressure Pmax', 'Pressure Angle θ_max',
               'Attitude Angle φ°', 'Cavitation Angle θ_cav']

def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*50}\n  {name} — Performance Metrics\n{'='*50}")
    rows = []
    for i, p in enumerate(PARAM_NAMES):
        yt, yp = y_true[:, i], y_pred[:, i]
        rms = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-10))) * 100
        rows.append({'Parameter': p, 'RMS Error': round(rms, 5), 'R²': round(r2, 5), 'MAPE (%)': round(mape, 3)})
        print(f"  {p:<22} | RMS={rms:.5f} | R²={r2:.4f} | MAPE={mape:.2f}%")
    return pd.DataFrame(rows).set_index('Parameter')

def regression_plot(y_true, y_pred, model_name):
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    for i, (ax, name) in enumerate(zip(axes.flatten(), PARAM_NAMES)):
        yt, yp = y_true[:, i], y_pred[:, i]
        ax.scatter(yt, yp, alpha=0.7, s=60, color='steelblue', edgecolors='navy', linewidths=0.5)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Y = T')
        ax.set_title(f'{name}\nR² = {r2_score(yt, yp):.4f} | MAPE = {np.mean(np.abs((yt - yp)/(np.abs(yt) + 1e-10)))*100:.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/regression_{model_name}.png', dpi=150)
    plt.close()

def plot_predictions(y_true, preds_dict, title_suffix=''):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    colors = {'FFNN': ('steelblue', 'o-', 2.0), 'RBNN': ('tomato', 's--', 1.5), 'GRNN': ('seagreen', '^:', 1.5)}
    x_idx = np.arange(len(y_true))
    for ax, i, name in zip(axes.flatten(), range(8), PARAM_NAMES):
        ax.plot(x_idx, y_true[:, i], 'ko-', lw=2.5, ms=7, label='Actual', zorder=5)
        for m_name, yp in preds_dict.items():
            col, style, lw = colors[m_name]
            ax.plot(x_idx, yp[:, i], style, color=col, lw=lw, ms=6, alpha=0.8, label=m_name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/comparison_{title_suffix}.png', dpi=150)
    plt.close()

def plot_error_comparison(m_ffnn, m_rbnn, m_grnn):
    params = PARAM_NAMES
    x = np.arange(len(params))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 8))
    
    fm = [float(m_ffnn.loc[p, 'MAPE (%)']) for p in params]
    rm = [float(m_rbnn.loc[p, 'MAPE (%)']) for p in params]
    gm = [float(m_grnn.loc[p, 'MAPE (%)']) for p in params]

    ax.bar(x - width, fm, width, label='FFNN', color='steelblue', edgecolor='navy')
    ax.bar(x,         rm, width, label='RBNN', color='tomato', edgecolor='darkred')
    ax.bar(x + width, gm, width, label='GRNN', color='seagreen', edgecolor='darkgreen')

    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('MAPE Error (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/plots/error_comparison.png', dpi=150)
    plt.close()

def time_model(model, X, model_type='keras'):
    t0 = time.perf_counter()
    if model_type == 'keras': model.predict(X, verbose=0)
    else: model.predict(X)
    return time.perf_counter() - t0