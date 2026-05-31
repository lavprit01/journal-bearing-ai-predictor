# src/data_prep.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath='data/table86_data.csv'):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, "
          f"{df.shape[1]} columns")
    return df

def explore_data(df):
    print("\n--- Data Summary ---")
    print(df[['LD','epsilon','S','RCf',
              'phi','Pmax']].describe().round(4))
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    os.makedirs('results/plots', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    outputs = ['S', 'RCf', 'phi', 'Pmax']
    titles  = ['Sommerfeld Number S',
               'Friction Variable f(R/C)',
               'Attitude Angle φ (°)',
               'Max Pressure Pmax']
    ld_vals = sorted(df['LD'].unique())
    colors  = plt.cm.tab10(
        np.linspace(0, 1, len(ld_vals))
    )
    for ax, col, title in zip(
        axes.flatten(), outputs, titles
    ):
        for ld, color in zip(ld_vals, colors):
            sub = df[df['LD'] == ld]
            ax.semilogy(
                sub['epsilon'], sub[col],
                marker='o', color=color,
                label=f'L/D={ld}', markersize=4
            )
        ax.set_xlabel('Eccentricity Ratio ε')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/01_data_overview.png',
                dpi=150)
    plt.close()
    print("EDA plot saved.")

def prepare_data(
    df,
    input_cols  = ['LD', 'epsilon'],
    output_cols = ['S', 'RCf', 'phi', 'Pmax'],
    test_size   = 0.15,
    val_size    = 0.15,
    random_seed = 42
):
    X = df[input_cols].values.astype(float)
    y = df[output_cols].values.astype(float)

    # Log10 transform skewed outputs
    y_t = y.copy()
    y_t[:, 0] = np.log10(y[:, 0] + 1e-10)  # S
    y_t[:, 1] = np.log10(y[:, 1] + 1e-10)  # RCf
    y_t[:, 2] = y[:, 2]                     # phi
    y_t[:, 3] = np.log10(y[:, 3] + 1e-10)  # Pmax

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y_t)

    total_test = test_size + val_size
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_s, y_s,
        test_size=total_test,
        random_state=random_seed
    )
    rel_val = val_size / total_test
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=(1 - rel_val),
        random_state=random_seed
    )

    print(f"\nData split:")
    print(f"  Train      : {len(X_tr)}")
    print(f"  Validation : {len(X_val)}")
    print(f"  Test       : {len(X_te)}")

    return (X_tr, X_val, X_te,
            y_tr, y_val, y_te,
            scaler_X, scaler_y)