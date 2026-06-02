# src/evaluate.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PARAM_NAMES = ['Sommerfeld S',
               'Leakage Flow QL',
               'Inlet Flow Qi',
               'Friction f(R/C)',
               'Max Pressure Pmax',
               'Pressure Angle θ_max',
               'Attitude Angle φ°',
               'Cavitation Angle θ_cav']


def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name} — Performance Metrics")
    print(f"{'='*50}")
    rows = []
    for i, p in enumerate(PARAM_NAMES):
        yt   = y_true[:, i]
        yp   = y_pred[:, i]
        rms  = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-10))) * 100
        rows.append({
            'Parameter': p,
            'RMS Error': round(rms,  5),
            'R²':        round(r2,   5),
            'MAPE (%)':  round(mape, 3)
        })
        print(f"  {p:<22} | RMS={rms:.5f} | R²={r2:.4f} | MAPE={mape:.2f}%")
    return pd.DataFrame(rows).set_index('Parameter')


def regression_plot(y_true, y_pred, model_name):
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    for i, (ax, name) in enumerate(zip(axes.flatten(), PARAM_NAMES)):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ax.scatter(yt, yp, alpha=0.7, s=60, color='steelblue', edgecolors='navy', linewidths=0.5)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Y = T')
        r2 = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-10))) * 100
        ax.set_title(f'{name}\nR² = {r2:.4f}  |  MAPE = {mape:.2f}%')
        ax.set_xlabel('Table 8.6 (Actual)')
        ax.set_ylabel(f'{model_name} (Predicted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{model_name} — Regression Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/plots/regression_{model_name}.png', dpi=150)
    plt.close()
    print(f"   Regression plot saved: {model_name}")


def plot_predictions(y_true, preds_dict, title_suffix=''):
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    colors = {
        'FFNN': ('steelblue', 'o-',  2.0),
        'RBNN': ('tomato',    's--', 1.5),
        'GRNN': ('seagreen',  '^:',  1.5)
    }
    x_idx = np.arange(len(y_true))

    for ax, i, name in zip(axes.flatten(), range(8), PARAM_NAMES):
        ax.plot(x_idx, y_true[:, i], 'ko-', lw=2.5, ms=7, label='Table 8.6 (Actual)', zorder=5)
        for model_name, yp in preds_dict.items():
            col, style, lw = colors[model_name]
            ax.plot(x_idx, yp[:, i], style, color=col, lw=lw, ms=6, alpha=0.8, label=model_name)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Table 8.6 vs All AI Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/plots/comparison_{title_suffix}.png', dpi=150)
    plt.close()
    print("   Comparison plot saved.")


def plot_error_comparison(m_ffnn, m_rbnn, m_grnn):
    """
    Bar chart comparing MAPE% of all 3 models for each output parameter
    """
    params = PARAM_NAMES
    x      = np.arange(len(params))
    width  = 0.25

    ffnn_mape = [float(m_ffnn.loc[p, 'MAPE (%)']) for p in params]
    rbnn_mape = [float(m_rbnn.loc[p, 'MAPE (%)']) for p in params]
    grnn_mape = [float(m_grnn.loc[p, 'MAPE (%)']) for p in params]

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width, ffnn_mape, width, label='FFNN', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x,         rbnn_mape, width, label='RBNN', color='tomato', edgecolor='darkred')
    bars3 = ax.bar(x + width, grnn_mape, width, label='GRNN', color='seagreen', edgecolor='darkgreen')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), 
                        xytext=(0, 3), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Performance Parameter', fontsize=12)
    ax.set_ylabel('MAPE Error (%)', fontsize=12)
    ax.set_title('Prediction Error Comparison\nFFNN vs RBNN vs GRNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    
    ax.set_xticklabels(params, fontsize=10, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(ffnn_mape), max(rbnn_mape), max(grnn_mape)) * 1.25)

    plt.tight_layout()
    plt.savefig('results/plots/error_comparison.png', dpi=150)
    plt.close()
    print("   Error comparison plot saved.")

    
    print("\n" + "="*70)
    print("  ERROR SUMMARY TABLE")
    print("="*70)
    print(f"{'Parameter':<25} {'FFNN':>10} {'RBNN':>10} {'GRNN':>10}")
    print("-" * 70)
    for p, fm, rm, gm in zip(params, ffnn_mape, rbnn_mape, grnn_mape):
        best = min(fm, rm, gm)
        fn = ('✅' if fm == best else '  ')
        rn = ('✅' if rm == best else '  ')
        gn = ('✅' if gm == best else '  ')
        print(f"{p:<25} {fm:>9.2f}%{fn} {rm:>9.2f}%{rn} {gm:>9.2f}%{gn}")
    print("="*70)


def time_model(model, X, model_type='keras'):
    t0 = time.perf_counter()
    if model_type == 'keras':
        model.predict(X, verbose=0)
    else:
        model.predict(X)
    return time.perf_counter() - t0