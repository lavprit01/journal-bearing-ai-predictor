import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
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

#  - - Session state initialization  - -
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'last_LD' not in st.session_state:
    st.session_state.last_LD = None
if 'last_eps' not in st.session_state:
    st.session_state.last_eps = None

def inverse_transform(y_scaled):
    y_raw       = scaler_y.inverse_transform(y_scaled)
    y_out       = y_raw.copy()
    y_out[:, 0] = 10 ** y_raw[:, 0]   # S
    y_out[:, 1] = 10 ** y_raw[:, 1]   # RCf
    y_out[:, 2] = y_raw[:, 2]         # phi
    y_out[:, 3] = 10 ** y_raw[:, 3]   # Pmax
    return y_out

def predict(LD_val, eps_val):
    X = scaler_X.transform([[LD_val, eps_val]])
    pf = inverse_transform(ffnn.predict(X))[0] if ffnn else np.zeros(4)
    pr = inverse_transform(rbnn.predict(X))[0]
    pg = inverse_transform(grnn.predict(X))[0]
    return pf, pr, pg

#  HEADER
st.title("⚙️ Plain Journal Bearing Performance Predictor")
st.markdown("""
**AI-powered prediction using Artificial Neural Networks**  
*Based on: Applied Tribology — Khonsari & Booser (Table 8.6)*  
*Project by: Lavprit Anand | MNNIT Allahabad Summer-Internship 2026*
""")

st.divider()

#  SIDEBAR — INPUTS
with st.sidebar:
    st.header("📥 Input Parameters")
    st.markdown("Adjust the bearing parameters below:")

    LD_val = st.slider(
        label="Aspect Ratio L/D",
        min_value=0.125,
        max_value=2.0,
        value=1.0,
        step=0.025,
        help="Length to Diameter ratio of the bearing"
    )

    eps_val = st.slider(
        label="Eccentricity Ratio ε",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Ratio of eccentricity to radial clearance"
    )

    model_choice = st.multiselect(
        "Select Models to Compare",
        ['FFNN', 'RBNN', 'GRNN'],
        default=['FFNN', 'RBNN', 'GRNN']
    )

    st.divider()
    predict_btn = st.button("🚀 Predict Performance",
                             use_container_width=True,
                             type="primary")

    st.divider()
    st.markdown("### 📐 Input Validity")
    if 0.125 <= LD_val <= 2.0:
        st.success("L/D ✅ Within training range")
    else:
        st.warning("L/D ⚠ Outside training range")

    if 0.05 <= eps_val <= 0.95:
        st.success("ε ✅ Within training range")
    else:
        st.warning("ε ⚠ Outside training range")

#  STORE PREDICTIONS IN SESSION STATE
if predict_btn:
    pf, pr, pg = predict(LD_val, eps_val)
    st.session_state.predictions = (pf, pr, pg)
    st.session_state.last_LD  = LD_val
    st.session_state.last_eps = eps_val

#  MAIN AREA - RESULTS (persists across re-runs)
if st.session_state.predictions is not None:

    pf, pr, pg   = st.session_state.predictions
    LD_used      = st.session_state.last_LD
    eps_used     = st.session_state.last_eps

    model_preds = {'FFNN': pf, 'RBNN': pr, 'GRNN': pg}
    param_keys  = ['S', 'RCf', 'phi', 'Pmax']
    param_names = ['Sommerfeld Number S',
                   'Friction Variable f(R/C)',
                   'Attitude Angle φ (°)',
                   'Max Pressure Pmax']

    st.subheader(f"📊 Results for L/D = {LD_used}, ε = {eps_used}")

    #  - - KPI Cards (FFNN results)  - -
    col1, col2, col3, col4 = st.columns(4)
    cards = [col1, col2, col3, col4]
    icons = ['📈', '🔧', '📐', '💥']
    units = ['—', '—', '°', '—']

    for col, icon, name, key, unit in zip(
        cards, icons, param_names, param_keys, units
    ):
        val = model_preds['FFNN'][param_keys.index(key)]
        col.metric(
            label=f"{icon} {key}",
            value=f"{val:.4f} {unit}",
            help=name
        )

    st.divider()

    #  - - Bar chart comparison  - -
    st.subheader("🔵 Model Comparison — All Parameters")

    fig_bar = go.Figure()
    colors  = {'FFNN': '#2196F3', 'RBNN': '#FF5722', 'GRNN': '#4CAF50'}

    for model_name in model_choice:
        vals = [model_preds[model_name][i] for i in range(4)]
        fig_bar.add_trace(go.Bar(
            name=model_name,
            x=param_names,
            y=vals,
            marker_color=colors[model_name],
            text=[f"{v:.4f}" for v in vals],
            textposition='outside'
        ))

    fig_bar.update_layout(
        barmode='group',
        title=f'Predicted Performance Parameters — L/D={LD_used}, ε={eps_used}',
        xaxis_title='Performance Parameter',
        yaxis_title='Predicted Value',
        legend_title='Model',
        height=450,
        template='plotly_white'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    #  - - Prediction Table  - -
    st.subheader("📋 Detailed Prediction Table")
    table_data = {'Parameter': param_names}
    for model_name in model_choice:
        table_data[model_name] = [
            f"{model_preds[model_name][i]:.5f}"
            for i in range(4)
        ]

    # Check against Table 8.6 if exact match exists
    match = df[
        (np.isclose(df['LD'],      LD_used,  atol=1e-3)) &
        (np.isclose(df['epsilon'], eps_used, atol=1e-3))
    ]
    if not match.empty:
        actual = match.iloc[0]
        table_data['Table 8.6 (Actual)'] = [
            f"{actual[k]:.5f}" for k in param_keys
        ]
        table_data['FFNN Error %'] = [
            f"{abs(pf[i] - actual[param_keys[i]]) / (abs(actual[param_keys[i]]) + 1e-10) * 100:.2f}%"
            for i in range(4)
        ]
        table_data['RBNN Error %'] = [
            f"{abs(pr[i] - actual[param_keys[i]]) / (abs(actual[param_keys[i]]) + 1e-10) * 100:.2f}%"
            for i in range(4)
        ]
        table_data['GRNN Error %'] = [
            f"{abs(pg[i] - actual[param_keys[i]]) / (abs(actual[param_keys[i]]) + 1e-10) * 100:.2f}%"
            for i in range(4)
        ]
        st.info("✅ Exact match found in Table 8.6 — showing validation error")

    # Show prediction table (NOT the raw df)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.divider()

    #  - - Parametric sweep plot  - -
    st.subheader("📉 Parametric Analysis — Vary ε at fixed L/D")

    selected_param = st.selectbox(
        "Select parameter to plot:",
        param_names,
        index=0
    )
    param_idx = param_names.index(selected_param)

    eps_range  = np.arange(0.05, 1.0, 0.05)
    sweep_data = {m: [] for m in ['FFNN', 'RBNN', 'GRNN']}

    for e in eps_range:
        pf_, pr_, pg_ = predict(LD_used, e)   # use LD_used (stored value)
        sweep_data['FFNN'].append(pf_[param_idx])
        sweep_data['RBNN'].append(pr_[param_idx])
        sweep_data['GRNN'].append(pg_[param_idx])

    fig_sweep = go.Figure()
    for model_name in model_choice:
        fig_sweep.add_trace(go.Scatter(
            x=eps_range,
            y=sweep_data[model_name],
            mode='lines+markers',
            name=model_name,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    # Add actual Table 8.6 data points
    subset = df[np.isclose(df['LD'], LD_used, atol=0.05)]
    if not subset.empty:
        fig_sweep.add_trace(go.Scatter(
            x=subset['epsilon'],
            y=subset[param_keys[param_idx]],
            mode='markers',
            name='Table 8.6 (Actual)',
            marker=dict(symbol='star', size=12, color='black')
        ))

    fig_sweep.update_layout(
        title=f'{selected_param} vs ε  (L/D = {LD_used})',
        xaxis_title='Eccentricity Ratio ε',
        yaxis_title=selected_param,
        template='plotly_white',
        height=400,
        legend_title='Source'
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    st.divider()

    #   Film thickness visualization 
    st.subheader("🔵 Oil Film Thickness Profile")

    theta     = np.linspace(0, 2 * np.pi, 360)
    h_theta   = 1 + eps_used * np.cos(theta)
    theta_deg = np.degrees(theta)

    fig_film = go.Figure()
    fig_film.add_trace(go.Scatter(
        x=theta_deg, y=h_theta,
        mode='lines',
        fill='tozeroy',
        name='Film Thickness h̄(θ)',
        line=dict(color='steelblue', width=2),
        fillcolor='rgba(70,130,180,0.3)'
    ))
    fig_film.add_hline(
        y=1 - eps_used,
        line_dash='dash',
        line_color='red',
        annotation_text=f'hmin = {1-eps_used:.3f}C'
    )
    fig_film.update_layout(
        title=f'Dimensionless Film Thickness — ε = {eps_used}',
        xaxis_title='Circumferential Angle θ (°)',
        yaxis_title='Dimensionless Film Thickness h̄ = h/C',
        template='plotly_white',
        height=350
    )
    st.plotly_chart(fig_film, use_container_width=True)

# If no prediction yet - show dataset overview 
else:
    st.info("👈 Set L/D and ε in the sidebar, then click **Predict Performance**")

    st.subheader("📚 Dataset Overview - Table 8.6 (Khonsari & Booser)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Points", len(df))
    col2.metric("L/D Values", df['LD'].nunique())
    col3.metric("ε Values per L/D", df['epsilon'].nunique())
    col4.metric("Output Parameters", "4")

    st.markdown(f"Showing all **{len(df)} rows** from Table 8.6")
    st.dataframe(
        df[['LD', 'epsilon', 'S', 'RCf', 'phi', 'Pmax']],
        use_container_width=True,
        hide_index=True,
        height=35 * len(df) + 40   # dynamic height — shows all 152 rows
    )

# Footer
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
Journal Bearing AI Predictor | 
Applied Tribology — Khonsari & Booser | 
MNNIT Allahabad Summer Internship 2026
</div>
""", unsafe_allow_html=True)