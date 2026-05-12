# ============================================================
#  Data Science Projects Dashboard – Enhanced UI
#  Project 1: Time Series Forecasting (Airline Passengers)
#  Project 2: Diabetes Classification (Random Forest)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ── App Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DS Projects Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp { background: #050A0E; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0A1628 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] .stRadio label {
    color: #7EB8F7 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 6px 0 !important;
}
[data-testid="stSidebar"] hr { border-color: #1E3A5F; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BFFF 0%, #5E6AD2 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    color: #4A7FA5;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8F4FD;
    margin: 2rem 0 0.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #0D1B2A 0%, #111D2E 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
    margin-bottom: 0.5rem;
}
.metric-card:hover { border-color: #38BFFF; transform: translateY(-2px); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38BFFF, #5E6AD2);
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4A7FA5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38BFFF;
}
.info-box {
    background: linear-gradient(135deg, #0D1B2A, #0A1628);
    border: 1px solid #1E3A5F;
    border-left: 3px solid #38BFFF;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #7EB8F7;
    font-size: 0.85rem;
    line-height: 1.7;
    margin: 1rem 0;
}
.proj-card {
    background: linear-gradient(135deg, #0D1B2A 0%, #0F2035 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.proj-card:hover {
    border-color: #38BFFF;
    box-shadow: 0 0 30px rgba(56, 191, 255, 0.08);
}
.proj-card h3 { font-family:'Syne',sans-serif !important; color:#E8F4FD !important; font-size:1.2rem !important; }
.proj-card p, .proj-card li { color: #7EB8F7; font-size: 0.85rem; line-height: 1.8; }
.tag {
    display: inline-block;
    background: #0D1B2A;
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #38BFFF;
    margin: 3px;
    font-family: 'DM Mono', monospace;
}
.accuracy-badge {
    background: linear-gradient(135deg, #064E3B, #065F46);
    border: 1px solid #10B981;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.acc-num { font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:#10B981; }
.acc-label { font-family:'DM Mono',monospace; font-size:0.75rem; color:#6EE7B7; text-transform:uppercase; letter-spacing:0.1em; }
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1E3A5F, transparent);
    margin: 2.5rem 0;
}
div[data-testid="stMetric"] {
    background: #0D1B2A;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 1rem;
}
div[data-testid="stMetric"] label { color: #4A7FA5 !important; font-size: 0.75rem !important; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #38BFFF !important; font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ─────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#050A0E",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#7EB8F7", family="DM Mono, monospace"),
    title_font=dict(color="#E8F4FD", family="Syne, sans-serif", size=16),
    xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#4A7FA5"),
    yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#4A7FA5"),
    legend=dict(bgcolor="#0D1B2A", bordercolor="#1E3A5F", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=20),
)

C = {
    "blue":   "#38BFFF",
    "indigo": "#5E6AD2",
    "purple": "#A78BFA",
    "green":  "#10B981",
    "orange": "#F59E0B",
    "red":    "#EF4444",
    "teal":   "#06B6D4",
    "muted":  "#4A7FA5",
}


# ============================================================
#  DATA LOADING
# ============================================================

@st.cache_data
def load_airline():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, header=0, names=["Month", "Passengers"])
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    return df.sort_index()


@st.cache_data
def load_diabetes():
    return pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


# ============================================================
#  PROJECT 1 LOGIC
# ============================================================

def compute_mas(df):
    df = df.copy()
    for w in [3, 6, 12]:
        df[f"SMA_{w}"] = df["Passengers"].rolling(window=w).mean()
        df[f"EMA_{w}"] = df["Passengers"].ewm(span=w, adjust=False).mean()
    return df


def mae(actual, predicted):
    idx = actual.index.intersection(predicted.dropna().index)
    return np.mean(np.abs(actual[idx] - predicted[idx]))


def compute_forecast(df):
    slope, *_ = linregress(np.arange(len(df)), df["Passengers"].values)
    drift = slope * 12
    future = pd.date_range("1961-01", periods=12, freq="MS")
    return (
        pd.Series(df["Passengers"].iloc[-12:].values + drift, index=future),
        pd.Series(df["EMA_12"].iloc[-12:].values + drift, index=future),
        drift,
    )


# ── Project 1 Charts ─────────────────────────────────────────

def chart_time_series(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Passengers"],
        fill="tozeroy", fillcolor="rgba(56,191,255,0.08)",
        line=dict(color=C["blue"], width=2.5), name="Monthly Passengers",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y}k passengers<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, title="Monthly Airline Passengers (1949–1960)",
                      xaxis_title="Year", yaxis_title="Passengers (thousands)", height=380)
    return fig


def chart_yearly_box(df):
    df2 = df.copy(); df2["Year"] = df2.index.year
    fig = go.Figure()
    for yr in sorted(df2["Year"].unique()):
        fig.add_trace(go.Box(
            y=df2[df2["Year"]==yr]["Passengers"], name=str(yr),
            marker_color=C["blue"], line_color=C["indigo"],
            fillcolor="rgba(56,191,255,0.15)", showlegend=False,
        ))
    fig.update_layout(**DARK_LAYOUT, title="Distribution by Year",
                      yaxis_title="Passengers (thousands)", height=380)
    return fig


def chart_sma(df, windows):
    palette = [C["red"], C["orange"], C["green"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Passengers"],
        line=dict(color=C["muted"], width=1.5, dash="dot"), opacity=0.7, name="Actual"))
    for w, c in zip(windows, palette):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{w}"],
            line=dict(color=c, width=2.5), name=f"SMA (w={w})",
            hovertemplate=f"SMA-{w}: %{{y:.1f}}k<extra></extra>"))
    fig.update_layout(**DARK_LAYOUT, title="Simple Moving Average (SMA)",
                      xaxis_title="Year", yaxis_title="Passengers (thousands)", height=400)
    return fig


def chart_ema(df, windows):
    palette = [C["purple"], C["indigo"], C["teal"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Passengers"],
        line=dict(color=C["muted"], width=1.5, dash="dot"), opacity=0.7, name="Actual"))
    for w, c in zip(windows, palette):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{w}"],
            line=dict(color=c, width=2.5), name=f"EMA (span={w})",
            hovertemplate=f"EMA-{w}: %{{y:.1f}}k<extra></extra>"))
    fig.update_layout(**DARK_LAYOUT, title="Exponential Moving Average (EMA)",
                      xaxis_title="Year", yaxis_title="Passengers (thousands)", height=400)
    return fig


def chart_sma_vs_ema(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Passengers"],
        line=dict(color=C["muted"], width=1.5), opacity=0.6, name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_12"],
        line=dict(color=C["green"], width=3), name="SMA-12"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_12"],
        line=dict(color=C["purple"], width=3, dash="dash"), name="EMA-12"))
    fig.update_layout(**DARK_LAYOUT, title="SMA vs EMA – 12-Month Window",
                      xaxis_title="Year", yaxis_title="Passengers (thousands)", height=400)
    return fig


def chart_forecast(df, sma_f, ema_f):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Passengers"],
        line=dict(color=C["muted"], width=1.5), opacity=0.7, name="Historical"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_12"],
        line=dict(color=C["green"], width=2), name="SMA-12 (hist)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_12"],
        line=dict(color=C["purple"], width=2, dash="dash"), name="EMA-12 (hist)"))
    fig.add_trace(go.Scatter(x=sma_f.index, y=sma_f.values,
        line=dict(color=C["green"], width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=8, color=C["green"]),
        name="SMA-12 Forecast 1961",
        hovertemplate="%{x|%b %Y}: %{y:.1f}k<extra>SMA Forecast</extra>"))
    fig.add_trace(go.Scatter(x=ema_f.index, y=ema_f.values,
        line=dict(color=C["purple"], width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=8, symbol="square", color=C["purple"]),
        name="EMA-12 Forecast 1961",
        hovertemplate="%{x|%b %Y}: %{y:.1f}k<extra>EMA Forecast</extra>"))
    fig.add_vline(x=pd.Timestamp("1961-01").timestamp()*1000,
        line_dash="dash", line_color=C["red"],
        annotation_text="Forecast Start", annotation_font_color=C["red"])
    fig.update_layout(**DARK_LAYOUT, title="Airline Passengers – 1961 Forecast",
                      xaxis_title="Year", yaxis_title="Passengers (thousands)", height=440)
    return fig


def chart_mae(mae_results):
    methods = list(mae_results.keys())
    errors  = list(mae_results.values())
    best    = min(mae_results, key=mae_results.get)
    colors  = [C["green"] if m == best else C["indigo"] for m in methods]
    fig = go.Figure(go.Bar(
        x=methods, y=errors,
        marker=dict(color=colors, line=dict(color="#1E3A5F", width=1)),
        text=[f"{v:.1f}" for v in errors], textposition="outside",
        textfont=dict(color="#E8F4FD", family="Syne, sans-serif", size=12),
        hovertemplate="%{x}: %{y:.2f}k<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, title="MAE Comparison – Lower is Better",
                      yaxis_title="Mean Absolute Error (thousands)",
                      height=380, showlegend=False)
    return fig


# ============================================================
#  PROJECT 2 LOGIC
# ============================================================

@st.cache_data
def train_model(df_raw):
    df = df_raw.copy()
    zeros_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zeros_cols] = df[zeros_cols].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    return (
        df, X, y, X_train, X_test,
        accuracy_score(y_test, y_pred),
        classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]),
        confusion_matrix(y_test, y_pred),
        pd.Series(model.feature_importances_, index=X.columns).sort_values(),
    )


# ── Project 2 Charts ─────────────────────────────────────────

def chart_cm(cm):
    labels = ["Not Diabetic", "Diabetic"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#0D1B2A"], [0.5, "#1E3A8A"], [1, C["blue"]]],
        text=cm, texttemplate="<b>%{text}</b>",
        textfont=dict(size=22, color="white", family="Syne, sans-serif"),
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual", height=360)
    return fig


def chart_importance(imp):
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker=dict(color=imp.values,
                    colorscale=[[0, C["indigo"]], [1, C["blue"]]],
                    line=dict(color="#1E3A5F", width=1)),
        text=[f"{v:.3f}" for v in imp.values], textposition="outside",
        textfont=dict(color="#E8F4FD", size=11),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, title="Feature Importances",
                      xaxis_title="Importance Score", height=400, showlegend=False)
    return fig


def chart_pie(y):
    counts = y.value_counts()
    fig = go.Figure(go.Pie(
        labels=["Not Diabetic", "Diabetic"], values=counts.values,
        hole=0.55,
        marker=dict(colors=[C["green"], C["red"]], line=dict(color="#050A0E", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, family="DM Mono, monospace"),
    ))
    fig.update_layout(**DARK_LAYOUT, title="Class Distribution", height=360, showlegend=False)
    return fig


def chart_feature_dist(df, feature):
    fig = go.Figure()
    for outcome, label, color in [(0, "Not Diabetic", C["green"]), (1, "Diabetic", C["red"])]:
        fig.add_trace(go.Histogram(
            x=df[df["Outcome"]==outcome][feature],
            name=label, opacity=0.75, marker_color=color, nbinsx=30,
            hovertemplate=f"{label}: %{{x}}<extra></extra>",
        ))
    fig.update_layout(**DARK_LAYOUT, title=f"Distribution of {feature} by Outcome",
                      xaxis_title=feature, yaxis_title="Count",
                      barmode="overlay", height=380)
    return fig


# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,#38BFFF,#A78BFA);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            DS Dashboard
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4A7FA5;
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">
            Data Science Projects
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["🏠  Home", "📈  Project 1 – Time Series", "🤖  Project 2 – Diabetes ML"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#4A7FA5;line-height:1.8;">
        <div>📚 Fundamentals of Data Science</div>
        <div>📅 Due: Week 14</div>
        <div style="margin-top:0.8rem;">
            <span class="tag">Python</span><span class="tag">Pandas</span>
            <span class="tag">Plotly</span><span class="tag">Sklearn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  HOME
# ============================================================

if page == "🏠  Home":
    st.markdown('<div class="hero-title">Data Science<br>Projects Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fundamentals of Data Science · Practical Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""<div class="proj-card">
            <h3>📈 Project 1: Time Series Forecasting</h3>
            <p>Analyze historical airline passenger data and predict future values using classical smoothing techniques.</p>
            <ul>
                <li>Simple Moving Average (SMA)</li>
                <li>Exponential Moving Average (EMA)</li>
                <li>MAE Error Evaluation</li>
                <li>1961 Passenger Forecast</li>
            </ul>
            <div style="margin-top:1rem;">
                <span class="tag">Airline Passengers</span>
                <span class="tag">1949–1960</span>
                <span class="tag">Forecasting</span>
            </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="proj-card">
            <h3>🤖 Project 2: Diabetes Classification</h3>
            <p>Build a machine learning model to classify diabetes patients using the Pima Indians dataset.</p>
            <ul>
                <li>Data Preprocessing & Cleaning</li>
                <li>Random Forest Classifier</li>
                <li>Confusion Matrix & Accuracy</li>
                <li>Feature Importance Analysis</li>
            </ul>
            <div style="margin-top:1rem;">
                <span class="tag">Pima Diabetes</span>
                <span class="tag">768 Records</span>
                <span class="tag">Classification</span>
            </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Quick Stats</div>', unsafe_allow_html=True)

    with st.spinner("Loading datasets..."):
        df_air = load_airline()
        df_dia = load_diabetes()

    for col, label, val in zip(
        st.columns(4),
        ["Airline Records", "Diabetes Records", "ML Features", "Years of Data"],
        [len(df_air), len(df_dia), df_dia.shape[1]-1, "12 years"],
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)


# ============================================================
#  PROJECT 1
# ============================================================

elif page == "📈  Project 1 – Time Series":
    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Time Series<br>Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Airline Passengers Dataset · 1949–1960</div>', unsafe_allow_html=True)

    with st.spinner("Loading & processing data..."):
        df_raw = load_airline()
        df = compute_mas(df_raw)
        sma_f, ema_f, drift = compute_forecast(df)

    for col, label, val in zip(
        st.columns(4),
        ["Total Records", "Min Passengers", "Max Passengers", "Mean Passengers"],
        [len(df_raw), f"{df_raw['Passengers'].min()}k", f"{df_raw['Passengers'].max()}k", f"{df_raw['Passengers'].mean():.1f}k"],
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 1. Dataset
    st.markdown('<div class="section-title">1 · Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(12), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 2. EDA
    st.markdown('<div class="section-title">2 · Exploratory Data Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(chart_time_series(df), use_container_width=True)
    col2.plotly_chart(chart_yearly_box(df), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 3. SMA
    st.markdown('<div class="section-title">3 · Simple Moving Average (SMA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">SMA smooths data by averaging a fixed number of past observations. A <b>larger window</b> = smoother line but slower to react. Click legend items to show/hide lines.</div>', unsafe_allow_html=True)
    windows = st.multiselect("Select SMA windows:", [3, 6, 12], default=[3, 6, 12], key="sma")
    if windows:
        st.plotly_chart(chart_sma(df, windows), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 4. EMA
    st.markdown('<div class="section-title">4 · Exponential Moving Average (EMA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">EMA gives <b>more weight to recent values</b>, making it more responsive than SMA. Toggle windows below to compare.</div>', unsafe_allow_html=True)
    spans = st.multiselect("Select EMA spans:", [3, 6, 12], default=[3, 6, 12], key="ema")
    if spans:
        st.plotly_chart(chart_ema(df, spans), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 5. SMA vs EMA
    st.markdown('<div class="section-title">5 · SMA vs EMA Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_sma_vs_ema(df), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 6. MAE
    st.markdown('<div class="section-title">6 · Error Evaluation (MAE)</div>', unsafe_allow_html=True)
    actual = df["Passengers"]
    mae_results = {f"SMA-{w}": mae(actual, df[f"SMA_{w}"]) for w in [3,6,12]}
    mae_results.update({f"EMA-{w}": mae(actual, df[f"EMA_{w}"]) for w in [3,6,12]})
    best = min(mae_results, key=mae_results.get)

    for col, (method, error) in zip(st.columns(6), mae_results.items()):
        is_best = method == best
        border = C["green"] if is_best else "#1E3A5F"
        vcolor = C["green"] if is_best else C["blue"]
        col.markdown(f"""<div class="metric-card" style="border-color:{border};">
            <div class="metric-label">{'⭐ ' if is_best else ''}{method}</div>
            <div class="metric-value" style="color:{vcolor};font-size:1.3rem;">{error:.2f}k</div>
            </div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_mae(mae_results), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 7. Forecast
    st.markdown('<div class="section-title">7 · Forecast for 1961</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Method: seasonal repeat of last 12 months + linear trend drift of <b>{drift:+.1f}k passengers/year</b>. Hover to explore values.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown(f"""<div class="metric-card">
        <div class="metric-label">SMA-12 Forecast Range</div>
        <div class="metric-value" style="font-size:1.2rem;">{sma_f.min():.0f}k – {sma_f.max():.0f}k</div></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <div class="metric-label">EMA-12 Forecast Range</div>
        <div class="metric-value" style="font-size:1.2rem;">{ema_f.min():.0f}k – {ema_f.max():.0f}k</div></div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_forecast(df, sma_f, ema_f), use_container_width=True)

    forecast_df = pd.DataFrame({
        "Month": sma_f.index.strftime("%b %Y"),
        "SMA-12 (k)": sma_f.round(1).values,
        "EMA-12 (k)": ema_f.round(1).values,
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# ============================================================
#  PROJECT 2
# ============================================================

elif page == "🤖  Project 2 – Diabetes ML":
    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Diabetes<br>Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Pima Indians Dataset · Random Forest Classifier</div>', unsafe_allow_html=True)

    with st.spinner("Training model..."):
        df_raw = load_diabetes()
        df, X, y, X_train, X_test, acc, report, cm, imp = train_model(df_raw)

    for col, label, val in zip(
        st.columns(4),
        ["Total Records", "Features", "Training Samples", "Test Samples"],
        [df_raw.shape[0], df_raw.shape[1]-1, X_train.shape[0], X_test.shape[0]],
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 1. Dataset
    st.markdown('<div class="section-title">1 · Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 2. Preprocessing
    st.markdown('<div class="section-title">2 · Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Columns like <b>Glucose</b>, <b>BloodPressure</b>, <b>SkinThickness</b>, <b>Insulin</b>, and <b>BMI</b>
        use <code>0</code> as a placeholder for missing values — biologically impossible.
        These are replaced with <code>NaN</code> then filled with the <b>column median</b>.
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 3. Performance
    st.markdown('<div class="section-title">3 · Model Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div class="accuracy-badge">
            <div class="acc-label">Model Accuracy</div>
            <div class="acc-num">{acc*100:.1f}%</div>
            <div class="acc-label" style="margin-top:0.5rem;">Random Forest · 100 Trees</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("**Classification Report**")
        st.code(report, language="text")
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 4. Visualizations
    st.markdown('<div class="section-title">4 · Visualizations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(chart_cm(cm), use_container_width=True)
    col2.plotly_chart(chart_pie(y), use_container_width=True)
    st.plotly_chart(chart_importance(imp), use_container_width=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # 5. Feature Explorer
    st.markdown('<div class="section-title">5 · Interactive Feature Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Select any feature to compare its distribution between diabetic and non-diabetic patients.</div>', unsafe_allow_html=True)
    feature = st.selectbox("Choose a feature:", X.columns.tolist())
    st.plotly_chart(chart_feature_dist(df, feature), use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <b>Key Takeaways:</b><br>
        • <b>Glucose</b> is the most important predictor of diabetes.<br>
        • The dataset is slightly imbalanced (~65% non-diabetic, ~35% diabetic).<br>
        • Random Forest achieves solid accuracy without any hyperparameter tuning.
    </div>""", unsafe_allow_html=True)