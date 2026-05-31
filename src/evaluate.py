# src/evaluate.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import (mean_squared_error,
                              r2_score,
                              mean_absolute_error)

PARAM_NAMES = ['Sommerfeld S',
               'Friction f(R/C)',
               'Attitude φ°',
               'Pmax']


def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*45}")
    print(f"  {name} — Performance Metrics")
    print(f"{'='*45}")
    rows = []
    for i, p in enumerate(PARAM_NAMES):
        yt   = y_true[:, i]
        yp   = y_pred[:, i]
        rms  = np.sqrt(
            mean_squared_error(yt, yp)
        )
        r2   = r2_score(yt, yp)
        mape = np.mean(
            np.abs((yt - yp) /
                   (np.abs(yt) + 1e-10))
        ) * 100
        rows.append({
            'Parameter': p,
            'RMS Error': round(rms,  5),
            'R²':        round(r2,   5),
            'MAPE (%)':  round(mape, 3)
        })
        print(f"  {p:<20} | RMS={rms:.5f} | "
              f"R²={r2:.4f} | MAPE={mape:.2f}%")
    return pd.DataFrame(rows).set_index(
        'Parameter'
    )


def regression_plot(y_true, y_pred, model_name):
    fig, axes = plt.subplots(2, 2,
                              figsize=(12, 10))
    for i, (ax, name) in enumerate(
        zip(axes.flatten(), PARAM_NAMES)
    ):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ax.scatter(yt, yp,
                   alpha=0.7, s=60,
                   color='steelblue',
                   edgecolors='navy',
                   linewidths=0.5)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi],
                'r--', lw=2, label='Y = T')
        r2 = r2_score(yt, yp)
        mape = np.mean(
            np.abs((yt - yp) /
                   (np.abs(yt) + 1e-10))
        ) * 100
        ax.set_title(
            f'{name}\n'
            f'R² = {r2:.4f}  |  '
            f'MAPE = {mape:.2f}%'
        )
        ax.set_xlabel('Table 8.6 (Actual)')
        ax.set_ylabel(f'{model_name} (Predicted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f'{model_name} — Regression Plot',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'results/plots/regression_{model_name}.png',
        dpi=150
    )
    plt.close()
    print(f"   Regression plot saved: {model_name}")


def plot_predictions(y_true, preds_dict,
                     title_suffix=''):
    fig, axes = plt.subplots(2, 2,
                              figsize=(14, 10))
    colors = {
        'FFNN': ('steelblue', 'o-',  2.0),
        'RBNN': ('tomato',    's--', 1.5),
        'GRNN': ('seagreen',  '^:',  1.5)
    }
    x_idx = np.arange(len(y_true))

    for ax, i, name in zip(
        axes.flatten(), range(4), PARAM_NAMES
    ):
        ax.plot(x_idx, y_true[:, i],
                'ko-', lw=2.5, ms=7,
                label='Table 8.6 (Actual)',
                zorder=5)
        for model_name, yp in preds_dict.items():
            col, style, lw = colors[model_name]
            ax.plot(x_idx, yp[:, i],
                    style, color=col,
                    lw=lw, ms=6,
                    alpha=0.8,
                    label=model_name)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Table 8.6 vs All AI Models',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'results/plots/'
        f'comparison_{title_suffix}.png',
        dpi=150
    )
    plt.close()
    print("   Comparison plot saved.")


def plot_error_comparison(m_ffnn, m_rbnn,
                           m_grnn):
    """
    Bar chart comparing MAPE% of all 3 models
    for each output parameter
    """
    params = PARAM_NAMES
    x      = np.arange(len(params))
    width  = 0.25

    ffnn_mape = [
        float(m_ffnn.loc[p, 'MAPE (%)'])
        for p in params
    ]
    rbnn_mape = [
        float(m_rbnn.loc[p, 'MAPE (%)'])
        for p in params
    ]
    grnn_mape = [
        float(m_grnn.loc[p, 'MAPE (%)'])
        for p in params
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, ffnn_mape,
                   width, label='FFNN',
                   color='steelblue',
                   edgecolor='navy')
    bars2 = ax.bar(x,         rbnn_mape,
                   width, label='RBNN',
                   color='tomato',
                   edgecolor='darkred')
    bars3 = ax.bar(x + width, grnn_mape,
                   width, label='GRNN',
                   color='seagreen',
                   edgecolor='darkgreen')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f'{h:.1f}%',
                xy=(bar.get_x()
                    + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9
            )

    ax.set_xlabel('Performance Parameter',
                  fontsize=12)
    ax.set_ylabel('MAPE Error (%)',
                  fontsize=12)
    ax.set_title(
        'Prediction Error Comparison\n'
        'FFNN vs RBNN vs GRNN',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(
        max(ffnn_mape),
        max(rbnn_mape),
        max(grnn_mape)
    ) * 1.25)

    plt.tight_layout()
    plt.savefig(
        'results/plots/error_comparison.png',
        dpi=150
    )
    plt.close()
    print("   Error comparison plot saved.")

    # Also print summary table
    print("\n" + "="*55)
    print("  ERROR SUMMARY TABLE")
    print("="*55)
    print(f"{'Parameter':<20} {'FFNN':>8} "
          f"{'RBNN':>8} {'GRNN':>8}")
    print("-"*55)
    for p, fm, rm, gm in zip(
        params, ffnn_mape, rbnn_mape, grnn_mape
    ):
        best = min(fm, rm, gm)
        fn = ('✅' if fm == best else '  ')
        rn = ('✅' if rm == best else '  ')
        gn = ('✅' if gm == best else '  ')
        print(f"{p:<20} "
              f"{fm:>7.2f}%{fn} "
              f"{rm:>7.2f}%{rn} "
              f"{gm:>7.2f}%{gn}")
    print("="*55)


def time_model(model, X, model_type='keras'):
    t0 = time.perf_counter()
    if model_type == 'keras':
        model.predict(X, verbose=0)
    else:
        model.predict(X)
    return time.perf_counter() - t0