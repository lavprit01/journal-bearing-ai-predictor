# src/test_single_input.py

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# 
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')
ffnn     = tf.keras.models.load_model(
               'models/ffnn_model.keras')
rbnn     = joblib.load('models/rbnn_model.pkl')
grnn     = joblib.load('models/grnn_model.pkl')
print("✅ All models loaded successfully!\n")

df = pd.read_csv('data/table86_data.csv')

PARAM_KEYS   = ['S', 'QL', 'Qi', 'RCf', 'Pmax', 'theta_max', 'phi', 'theta_cav']
PARAM_LABELS = ['S  (Sommerfeld)',
                'QL (Leakage Flow)',
                'Qi (Inlet Flow)',
                'RCf (Friction Var)',
                'Pmax (Max Press)',
                'θ_max (Peak Press °)',
                'φ (Attitude °)',
                'θ_cav (Cavitation °)']

def inverse_transform(y_scaled):
    y_raw       = scaler_y.inverse_transform(y_scaled)
    y_out       = y_raw.copy()
    
    # Reverse log10 ONLY for the variables we transformed in data_prep.py
    y_out[:, 0] = 10 ** y_raw[:, 0]  # S
    y_out[:, 1] = y_raw[:, 1]        # QL (Linear)
    y_out[:, 2] = y_raw[:, 2]        # Qi (Linear)
    y_out[:, 3] = 10 ** y_raw[:, 3]  # RCf
    y_out[:, 4] = 10 ** y_raw[:, 4]  # Pmax
    y_out[:, 5] = y_raw[:, 5]        # theta_max (Linear)
    y_out[:, 6] = y_raw[:, 6]        # phi (Linear)
    y_out[:, 7] = y_raw[:, 7]        # theta_cav (Linear)
    return y_out


def predict_all(LD_val, eps_val):
    X_raw    = np.array([[LD_val, eps_val]])
    X_scaled = scaler_X.transform(X_raw)
    pf = inverse_transform(
        ffnn.predict(X_scaled, verbose=0))[0]
    pr = inverse_transform(
        rbnn.predict(X_scaled))[0]
    pg = inverse_transform(
        grnn.predict(X_scaled))[0]
    return pf, pr, pg


def show_result(LD_val, eps_val):
    print("\n" + "="*65)
    print(f"  INPUT → L/D = {LD_val}  |  "
          f"ε = {eps_val}")
    print("="*65)

    if not (0.125 <= LD_val <= 2.0):
        print("⚠ L/D outside training range")
    if not (0.05 <= eps_val <= 0.95):
        print("⚠ ε outside training range")

    pf, pr, pg = predict_all(LD_val, eps_val)

    
    match = df[
        (np.isclose(df['LD'],
                    LD_val,  atol=1e-3)) &
        (np.isclose(df['epsilon'],
                    eps_val, atol=1e-3))
    ]
    has_actual = not match.empty

    if has_actual:
        actual = match.iloc[0]
        print(f"\n{'Parameter':<22} "
              f"{'Actual':>10} "
              f"{'FFNN':>10} {'Err%':>7} "
              f"{'RBNN':>10} {'Err%':>7} "
              f"{'GRNN':>10} {'Err%':>7}")
        print("-"*85)

        for i, (key, label) in enumerate(
            zip(PARAM_KEYS, PARAM_LABELS)
        ):
            av  = actual[key]
            fv  = pf[i]
            rv  = pr[i]
            gv  = pg[i]
            fe  = abs(fv-av)/(abs(av)+1e-10)*100
            re  = abs(rv-av)/(abs(av)+1e-10)*100
            ge  = abs(gv-av)/(abs(av)+1e-10)*100

            # Best model indicator
            best = min(fe, re, ge)
            fi = '✅' if fe == best else '  '
            ri = '✅' if re == best else '  '
            gi = '✅' if ge == best else '  '

            print(
                f"{label:<22} "
                f"{av:>10.4f} "
                f"{fv:>10.4f} {fe:>6.2f}%{fi} "
                f"{rv:>10.4f} {re:>6.2f}%{ri} "
                f"{gv:>10.4f} {ge:>6.2f}%{gi}"
            )
        print("-"*85)

        
        all_fe = [
            abs(pf[i] - actual[k]) /
            (abs(actual[k]) + 1e-10) * 100
            for i, k in enumerate(PARAM_KEYS)
        ]
        all_re = [
            abs(pr[i] - actual[k]) /
            (abs(actual[k]) + 1e-10) * 100
            for i, k in enumerate(PARAM_KEYS)
        ]
        all_ge = [
            abs(pg[i] - actual[k]) /
            (abs(actual[k]) + 1e-10) * 100
            for i, k in enumerate(PARAM_KEYS)
        ]
        avg_f = np.mean(all_fe)
        avg_r = np.mean(all_re)
        avg_g = np.mean(all_ge)

        print(f"\n  Average Error → "
              f"FFNN: {avg_f:.2f}%  "
              f"RBNN: {avg_r:.2f}%  "
              f"GRNN: {avg_g:.2f}%")

        best_overall = min(avg_f, avg_r, avg_g)
        if best_overall == avg_f:
            print("  🏆 Best model: FFNN")
        elif best_overall == avg_r:
            print("  🏆 Best model: RBNN")
        else:
            print("  🏆 Best model: GRNN")

    else:
        print(
            f"\n{'Parameter':<22} "
            f"{'FFNN':>12} "
            f"{'RBNN':>12} "
            f"{'GRNN':>12}"
        )
        print("-"*62)
        for i, (key, label) in enumerate(
            zip(PARAM_KEYS, PARAM_LABELS)
        ):
            print(
                f"{label:<22} "
                f"{pf[i]:>12.4f} "
                f"{pr[i]:>12.4f} "
                f"{pg[i]:>12.4f}"
            )
        print("-"*62)
        print("ℹ️  Unseen input — no Table 8.6 "
              "comparison available")


if __name__ == '__main__':

    print("🔵"*30)
    print("  JOURNAL BEARING AI PREDICTOR")
    print("  Models: FFNN | RBNN | GRNN")
    print("🔵"*30)

    # Known test points from Table 8.6
    print("\n[TEST 1] L/D=1.0, ε=0.50")
    show_result(1.0, 0.50)

    print("\n[TEST 2] L/D=0.5, ε=0.70")
    show_result(0.5, 0.70)

    print("\n[TEST 3] L/D=2.0, ε=0.30")
    show_result(2.0, 0.30)

    print("\n[TEST 4] L/D=0.75, ε=0.85")
    show_result(0.75, 0.85)

    # Unseen interpolation test
    print("\n[TEST 5] UNSEEN L/D=0.6, ε=0.45")
    show_result(0.6, 0.45)

    print("\n[TEST 6] UNSEEN L/D=1.2, ε=0.65")
    show_result(1.2, 0.65)

    # Custom input
    print("\n" + "="*65)
    print("[TEST 7] ENTER YOUR OWN VALUES")
    print("="*65)
    try:
        ld  = float(input(
            "  Enter L/D (0.125 to 2.0) : "))
        eps = float(input(
            "  Enter ε   (0.05  to 0.95): "))
        show_result(ld, eps)
    except (ValueError, KeyboardInterrupt):
        print("  Skipped.")

    print("\n" + "="*65)
    print("✅ All tests complete!")
    print("="*65)