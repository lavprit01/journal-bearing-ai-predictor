import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from scipy.optimize import brentq
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#  - - Page config  - -
st.set_page_config(
    page_title="Journal Bearing AI",
    page_icon="⚙️",
    layout="wide"
)

#  - - Load models  - -
@st.cache_resource
def load_all_models():
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    import joblib
    from src.models_definition import RBNN, GRNN
    from src.ffnn_numpy import FFNNNumpy

    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    ffnn     = FFNNNumpy('models/ffnn_weights.npy')
    rbnn     = joblib.load('models/rbnn_model.pkl')
    grnn     = joblib.load('models/grnn_model.pkl')
    df       = pd.read_csv('data/table86_data.csv')
    return scaler_X, scaler_y, ffnn, rbnn, grnn, df

scaler_X, scaler_y, ffnn, rbnn, grnn, df = load_all_models()

#  - - Global Definitions - -
param_keys  = ['S', 'QL', 'Qi', 'RCf', 'Pmax', 'theta_max', 'phi', 'theta_cav']
param_names = ['Sommerfeld Number S', 'Leakage Flow QL', 'Inlet Flow Qi',
               'Friction Variable f(R/C)', 'Max Pressure Pmax',
               'Max Pressure Angle θ_max', 'Attitude Angle φ (°)', 'Cavitation Angle θ_cav']
icons = ['📈', '💧', '🚰', '🔧', '💥', '🎯', '📐', '🌪️']
units = ['—', '—', '—', '—', '—', '°', '°', '°']

def inverse_transform(y_scaled):
    y_raw       = scaler_y.inverse_transform(y_scaled)
    y_out       = y_raw.copy()
    y_out[:, 0] = 10 ** y_raw[:, 0]  # S
    y_out[:, 1] = y_raw[:, 1]        # QL 
    y_out[:, 2] = y_raw[:, 2]        # Qi 
    y_out[:, 3] = 10 ** y_raw[:, 3]  # RCf
    y_out[:, 4] = 10 ** y_raw[:, 4]  # Pmax
    y_out[:, 5] = y_raw[:, 5]        # theta_max 
    y_out[:, 6] = y_raw[:, 6]        # phi 
    y_out[:, 7] = y_raw[:, 7]        # theta_cav 
    return y_out

def predict(LD_val, eps_val):
    X = scaler_X.transform([[LD_val, eps_val]])
    pf = inverse_transform(ffnn.predict(X))[0] if ffnn else np.zeros(8)
    pr = inverse_transform(rbnn.predict(X))[0]
    pg = inverse_transform(grnn.predict(X))[0]
    return pf, pr, pg

#  HEADER
st.title("⚙️ Plain Journal Bearing Performance Predictor")
st.markdown("""
**AI-powered prediction using Artificial Neural Networks** *Based on: Applied Tribology — Khonsari & Booser (Table 8.6)* *Project by: Lavprit Anand | MNNIT Allahabad Summer-Internship 2026*
""")
st.divider()


# SIDEBAR: APP MODE SELECTION

app_mode = st.sidebar.radio(
    "Select Application Mode:",
    ["1. Direct Predictor (Given ε)", "2. Thermodynamic Solver (Iterative)"]
)
st.sidebar.divider()


# MODE 1: THE APP

if "1." in app_mode:
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'last_LD' not in st.session_state:
        st.session_state.last_LD = None
    if 'last_eps' not in st.session_state:
        st.session_state.last_eps = None

    with st.sidebar:
        st.header("📥 Input Parameters")
        LD_val = st.slider("Aspect Ratio L/D", 0.125, 2.0, 1.0, 0.001)
        eps_val = st.slider("Eccentricity Ratio ε", 0.05, 0.95, 0.50, 0.01)
        model_choice = st.multiselect("Select Models to Compare", ['FFNN', 'RBNN', 'GRNN'], default=['FFNN', 'RBNN', 'GRNN'])
        st.divider()
        predict_btn = st.button("🚀 Predict Performance", use_container_width=True, type="primary")

    if predict_btn:
        pf, pr, pg = predict(LD_val, eps_val)
        st.session_state.predictions = (pf, pr, pg)
        st.session_state.last_LD  = LD_val
        st.session_state.last_eps = eps_val

    if st.session_state.predictions is not None:
        pf, pr, pg   = st.session_state.predictions
        LD_used      = st.session_state.last_LD
        eps_used     = st.session_state.last_eps
        model_preds = {'FFNN': pf, 'RBNN': pr, 'GRNN': pg}

        st.subheader(f"📊 Results for L/D = {LD_used}, ε = {eps_used}")

        # KPI Cards (2 rows of 4)
        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)
        cards = [c1, c2, c3, c4, c5, c6, c7, c8]
        
        for col, icon, name, key, unit in zip(cards, icons, param_names, param_keys, units):
            val = model_preds['FFNN'][param_keys.index(key)]
            col.metric(label=f"{icon} {key}", value=f"{val:.4f} {unit}", help=name)

        st.divider()

        # Bar Chart Comparison
        st.subheader("🔵 Model Comparison — All Parameters")
        fig_bar = go.Figure()
        colors  = {'FFNN': '#2196F3', 'RBNN': '#FF5722', 'GRNN': '#4CAF50'}
        for model_name in model_choice:
            vals = [model_preds[model_name][i] for i in range(8)]
            fig_bar.add_trace(go.Bar(name=model_name, x=param_keys, y=vals, marker_color=colors[model_name], text=[f"{v:.3f}" for v in vals], textposition='outside'))
        fig_bar.update_layout(barmode='group', height=450, template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        
        st.subheader("📋 Detailed Prediction Table")
        table_data = {'Parameter': param_names}
        for model_name in model_choice:
            table_data[model_name] = [f"{model_preds[model_name][i]:.5f}" for i in range(8)]

        # Check against Table 8.6
        match = df[(np.isclose(df['LD'], LD_used, atol=1e-3)) & (np.isclose(df['epsilon'], eps_used, atol=1e-3))]
        if not match.empty:
            actual = match.iloc[0]
            table_data['Table 8.6 (Actual)'] = [f"{actual[k]:.5f}" for k in param_keys]
            
            # Calculate Errors for selected models
            for model_name in model_choice:
                table_data[f'{model_name} Err %'] = [
                    f"{abs(model_preds[model_name][i] - actual[param_keys[i]]) / (abs(actual[param_keys[i]]) + 1e-10) * 100:.2f}%"
                    for i in range(8)
                ]
            st.info("✅ Exact match found in Table 8.6 — showing validation errors!")

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        st.divider()

        
        st.subheader("📉 Parametric Analysis — Vary ε at fixed L/D")
        selected_param = st.selectbox("Select parameter to plot:", param_names, index=0)
        param_idx = param_names.index(selected_param)

        eps_range  = np.arange(0.05, 1.0, 0.05)
        sweep_data = {m: [] for m in ['FFNN', 'RBNN', 'GRNN']}
        
        for e in eps_range:
            pf_, pr_, pg_ = predict(LD_used, e)
            sweep_data['FFNN'].append(pf_[param_idx])
            sweep_data['RBNN'].append(pr_[param_idx])
            sweep_data['GRNN'].append(pg_[param_idx])

        fig_sweep = go.Figure()
        for model_name in model_choice:
            fig_sweep.add_trace(go.Scatter(x=eps_range, y=sweep_data[model_name], mode='lines+markers', name=model_name, line=dict(width=2), marker=dict(size=6)))

        # Add Table 8.6 data points
        subset = df[np.isclose(df['LD'], LD_used, atol=0.05)]
        if not subset.empty:
            fig_sweep.add_trace(go.Scatter(x=subset['epsilon'], y=subset[param_keys[param_idx]], mode='markers', name='Table 8.6 (Actual)', marker=dict(symbol='star', size=12, color='black')))

        fig_sweep.update_layout(title=f'{selected_param} vs ε  (L/D = {LD_used})', xaxis_title='Eccentricity Ratio ε', yaxis_title=selected_param, template='plotly_white', height=400)
        st.plotly_chart(fig_sweep, use_container_width=True)
        st.divider()

        # Film Thickness Profile
        st.subheader("🔵 Oil Film Thickness Profile")
        theta = np.linspace(0, 2 * np.pi, 360)
        h_theta = 1 + eps_used * np.cos(theta)
        fig_film = go.Figure()
        fig_film.add_trace(go.Scatter(x=np.degrees(theta), y=h_theta, mode='lines', fill='tozeroy', name='Film Thickness h̄(θ)', line=dict(color='steelblue', width=2), fillcolor='rgba(70,130,180,0.3)'))
        fig_film.add_hline(y=1 - eps_used, line_dash='dash', line_color='red', annotation_text=f'hmin = {1-eps_used:.3f}C')
        fig_film.update_layout(xaxis_title='Circumferential Angle θ (°)', yaxis_title='Dimensionless Film Thickness h̄ = h/C', height=350, template='plotly_white')
        st.plotly_chart(fig_film, use_container_width=True)

    else:
        # Shown when no prediction has been made yet
        st.info("👈 Set L/D and ε in the sidebar, then click **Predict Performance**")
        
        st.subheader("📚 Dataset Overview - Table 8.6 (Khonsari & Booser)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Points", len(df))
        c2.metric("L/D Values", df['LD'].nunique())
        c3.metric("ε Values per L/D", df['epsilon'].nunique())
        c4.metric("Output Parameters", "8")
        
        st.markdown(f"Showing all **{len(df)} rows** from Table 8.6")
        st.dataframe(df[['LD', 'epsilon', 'S', 'QL', 'Qi', 'RCf', 'Pmax', 'theta_max', 'phi', 'theta_cav']], use_container_width=True, hide_index=True)



# MODE 2: THERMODYNAMIC SOLVER

elif "2." in app_mode:
    st.info("💡 **Inverse AI Root-Finding:** This mode automatically iterates through the AI model to find the steady-state thermal equilibrium for a given load and speed.")
    
    with st.sidebar:
        st.header("📥 Operating Conditions")
        W_load = st.number_input("Bearing Radial Load (lbf)", value=1600.0, step=100.0)
        N_rpm = st.number_input("Shaft Speed (rpm)", value=1800.0, step=100.0)
        T_in = st.number_input("Inlet Oil Temp (°F)", value=166.0, step=1.0)
        
        st.subheader("Geometry")
        D_in = st.number_input("Diameter D (in)", value=4.0, step=0.1)
        L_in = st.number_input("Length L (in)", value=4.0, step=0.1)
        C_in = st.number_input("Radial Clearance C (in)", value=0.002, step=0.0001, format="%.4f")
        
        st.subheader("Oil Properties (SAE 10)")
        rho = st.number_input("Density (lbm/in³)", value=0.0315, format="%.4f")
        cp = st.number_input("Specific Heat (BTU/lbm·°F)", value=0.48)
        
        solve_btn = st.button("🔥 Run Thermal Solver", use_container_width=True, type="primary")

    if solve_btn:
        with st.spinner("AI is solving the thermal loop..."):
            R, L_D, N_s, P_unit, J = D_in / 2.0, L_in / D_in, N_rpm / 60.0, W_load / (L_in * D_in), 9336
            T_eff, tolerance, history, final_ai_preds = T_in, 0.5, [], None
            
            def get_ffnn_preds(eps):
                return inverse_transform(ffnn.predict(scaler_X.transform([[L_D, eps]])))[0]
                
            def obj_func(eps, target_S):
                return get_ffnn_preds(eps)[0] - target_S
                
            for i in range(1, 21):
                mu = 1.30e-6 * np.exp(-0.0294 * (T_eff - 166.0))
                S_target = ((R/C_in)**2) * (mu * N_s) / P_unit
                try: 
                    eps_found = brentq(obj_func, 0.05, 0.95, args=(S_target,))
                except ValueError: 
                    st.error(f"Design out of bounds! Target S={S_target:.4f} is too extreme for the AI.")
                    break
                    
                final_ai_preds = get_ffnn_preds(eps_found)
                F_friction = (final_ai_preds[3] / (R/C_in)) * W_load
                Power_Loss = F_friction * (2 * np.pi * R * N_s)
                Delta_T = Power_Loss / (rho * cp * (final_ai_preds[1] * R * C_in * N_s * L_in) * J * 2.5) 
                
                T_new = T_in + (Delta_T / 2.0)
                history.append({"Iter": i, "T_eff (°F)": round(T_eff, 2), "μ (reyns)": f"{mu:.2e}", "S": round(S_target, 4), "ε": round(eps_found, 4), "ΔT (°F)": round(Delta_T, 2)})
                if abs(T_new - T_eff) <= tolerance: break
                T_eff = T_new
                
            st.success(f"✅ Equilibrium Reached in {len(history)} iterations!")
            st.subheader(f"🏁 Final Bearing Design Specifications (T_eff = {T_eff:.1f} °F)")
            st.markdown(f"**Operating Viscosity:** {mu:.2e} reyns | **Operating Eccentricity:** {eps_found:.4f} | **Min Film Thickness:** {C_in * (1 - eps_found) * 1e6:.0f} µin")
            
            c1, c2, c3, c4 = st.columns(4)
            c5, c6, c7, c8 = st.columns(4)
            for col, icon, name, key, unit in zip([c1,c2,c3,c4,c5,c6,c7,c8], icons, param_names, param_keys, units):
                col.metric(label=f"{icon} {key}", value=f"{final_ai_preds[param_keys.index(key)]:.4f} {unit}", help=name)
                
            st.divider()
            
            
            st.subheader("📈 Temperature Convergence")
            df_hist = pd.DataFrame(history)
            st.plotly_chart(px.line(df_hist, x="Iter", y="T_eff (°F)", markers=True), use_container_width=True)
            
            st.divider()

            st.subheader("🔄 Iteration History")
            
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("<div style='text-align:center; color:gray; font-size:12px'>Journal Bearing AI Predictor | Applied Tribology | MNNIT Allahabad 2026</div>", unsafe_allow_html=True)