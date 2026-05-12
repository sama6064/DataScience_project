# ============================================================
#  Data Science Projects Dashboard – Streamlit App
#  Project 1: Time Series Forecasting (Airline Passengers)
#  Project 2: Diabetes Classification (Random Forest)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")

# ── App Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DS Projects Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Global Plot Style ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

# ============================================================
#  DATA LOADING (cached so it only runs once)
# ============================================================

@st.cache_data
def load_airline_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, header=0, names=["Month", "Passengers"])
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    df = df.sort_index()
    return df


@st.cache_data
def load_diabetes_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df


# ============================================================
#  PROJECT 1 HELPERS
# ============================================================

def compute_moving_averages(df):
    df = df.copy()
    df["SMA_3"]  = df["Passengers"].rolling(window=3).mean()
    df["SMA_6"]  = df["Passengers"].rolling(window=6).mean()
    df["SMA_12"] = df["Passengers"].rolling(window=12).mean()
    df["EMA_3"]  = df["Passengers"].ewm(span=3,  adjust=False).mean()
    df["EMA_6"]  = df["Passengers"].ewm(span=6,  adjust=False).mean()
    df["EMA_12"] = df["Passengers"].ewm(span=12, adjust=False).mean()
    return df


def mae(actual, predicted):
    common = actual.index.intersection(predicted.dropna().index)
    return np.mean(np.abs(actual[common] - predicted[common]))


def compute_forecast(df):
    x = np.arange(len(df))
    slope, *_ = linregress(x, df["Passengers"].values)
    drift = slope * 12
    last_12 = df["Passengers"].iloc[-12:].values
    future_dates = pd.date_range(start="1961-01", periods=12, freq="MS")
    sma_forecast = pd.Series(last_12 + drift, index=future_dates)
    ema_forecast = pd.Series(df["EMA_12"].iloc[-12:].values + drift, index=future_dates)
    return sma_forecast, ema_forecast, drift


def plot_time_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(df.index, df["Passengers"], color="#2196F3", linewidth=2, label="Monthly Passengers")
    axes[0].fill_between(df.index, df["Passengers"], alpha=0.15, color="#2196F3")
    axes[0].set_title("Monthly Airline Passengers (1949–1960)")
    axes[0].set_ylabel("Passengers (thousands)")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend()

    df_tmp = df.copy()
    df_tmp["Year"] = df_tmp.index.year
    years  = sorted(df_tmp["Year"].unique())
    yearly = [df_tmp[df_tmp["Year"] == y]["Passengers"].values for y in years]
    bp = axes[1].boxplot(yearly, labels=years, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
    axes[1].set_title("Passenger Distribution by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Passengers (thousands)")

    plt.tight_layout()
    return fig


def plot_sma(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Passengers"], color="#90A4AE", linewidth=1.5, alpha=0.8, label="Actual")
    ax.plot(df.index, df["SMA_3"],  color="#F44336", linewidth=2,   linestyle="--", label="SMA (window=3)")
    ax.plot(df.index, df["SMA_6"],  color="#FF9800", linewidth=2,   linestyle="--", label="SMA (window=6)")
    ax.plot(df.index, df["SMA_12"], color="#4CAF50", linewidth=2.5, label="SMA (window=12)")
    ax.set_title("Simple Moving Average (SMA)")
    ax.set_xlabel("Year"); ax.set_ylabel("Passengers (thousands)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); plt.tight_layout()
    return fig


def plot_ema(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Passengers"], color="#90A4AE", linewidth=1.5, alpha=0.8, label="Actual")
    ax.plot(df.index, df["EMA_3"],  color="#E91E63", linewidth=2,   linestyle="--", label="EMA (span=3)")
    ax.plot(df.index, df["EMA_6"],  color="#9C27B0", linewidth=2,   linestyle="--", label="EMA (span=6)")
    ax.plot(df.index, df["EMA_12"], color="#3F51B5", linewidth=2.5, label="EMA (span=12)")
    ax.set_title("Exponential Moving Average (EMA)")
    ax.set_xlabel("Year"); ax.set_ylabel("Passengers (thousands)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); plt.tight_layout()
    return fig


def plot_sma_vs_ema(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Passengers"], color="#B0BEC5", linewidth=1.5, alpha=0.9, label="Actual")
    ax.plot(df.index, df["SMA_12"], color="#4CAF50", linewidth=2.5, label="SMA (window=12)")
    ax.plot(df.index, df["EMA_12"], color="#3F51B5", linewidth=2.5, linestyle="--", label="EMA (span=12)")
    ax.set_title("SMA vs EMA Comparison (12-month window)")
    ax.set_xlabel("Year"); ax.set_ylabel("Passengers (thousands)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); plt.tight_layout()
    return fig


def plot_forecast(df, sma_forecast, ema_forecast):
    future_dates = sma_forecast.index
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Passengers"], color="#607D8B", linewidth=1.5, alpha=0.8, label="Historical")
    ax.plot(df.index, df["SMA_12"], color="#4CAF50", linewidth=2, label="SMA-12 (historical)")
    ax.plot(df.index, df["EMA_12"], color="#3F51B5", linewidth=2, linestyle="--", label="EMA-12 (historical)")
    ax.plot(future_dates, sma_forecast, color="#4CAF50", linewidth=2.5, linestyle=":", marker="o", markersize=5, label="SMA-12 Forecast (1961)")
    ax.plot(future_dates, ema_forecast, color="#3F51B5", linewidth=2.5, linestyle=":", marker="s", markersize=5, label="EMA-12 Forecast (1961)")
    ax.axvline(pd.Timestamp("1961-01"), color="red", linestyle="--", alpha=0.6, label="Forecast Start")
    ax.set_title("Airline Passengers – Forecast for 1961")
    ax.set_xlabel("Year"); ax.set_ylabel("Passengers (thousands)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(); plt.tight_layout()
    return fig


def plot_mae(mae_results):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["#EF9A9A", "#EF5350", "#C62828", "#90CAF9", "#1E88E5", "#0D47A1"]
    methods = list(mae_results.keys())
    errors  = list(mae_results.values())
    bars = ax.bar(methods, errors, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("MAE Comparison – SMA vs EMA")
    ax.set_ylabel("Mean Absolute Error (thousands)")
    ax.set_ylim(0, max(errors) * 1.25)
    plt.tight_layout()
    return fig


# ============================================================
#  PROJECT 2 HELPERS
# ============================================================

@st.cache_data
def preprocess_and_train(df_raw):
    df = df_raw.copy()
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"])
    cm     = confusion_matrix(y_test, y_pred)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    return df, X, y, X_train, X_test, acc, report, cm, importances


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Diabetic", "Diabetic"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances):
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts()
    ax.pie(counts, labels=["Not Diabetic", "Diabetic"],
           autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    ax.set_title("Dataset Class Distribution")
    plt.tight_layout()
    return fig


# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 DS Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📈 Project 1 – Time Series", "🤖 Project 2 – Diabetes Classification"],
)
st.sidebar.markdown("---")



# ============================================================
#  HOME PAGE
# ============================================================

if page == "🏠 Home":
    st.title("📊 Data Science Projects Dashboard")
    st.markdown("### Fundamentals of Data Science – Practical Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Project 1: Time Series Forecasting")
        st.markdown("""
        Analyze historical airline passenger data and forecast future values using:
        - **Simple Moving Average (SMA)**
        - **Exponential Moving Average (EMA)**
        - **MAE error evaluation**
        
        **Dataset:** Airline Passengers (1949–1960)
        """)

    with col2:
        st.markdown("#### 🤖 Project 2: Diabetes Classification")
        st.markdown("""
        Build a machine learning model to predict diabetes using:
        - **Random Forest Classifier**
        - **Feature importance analysis**
        - **Confusion matrix & accuracy**
        
        **Dataset:** Pima Indians Diabetes Dataset
        """)

    st.markdown("---")
    st.markdown("#### 🛠️ Technologies Used")
    cols = st.columns(5)
    techs = ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib"]
    for col, tech in zip(cols, techs):
        col.success(tech)

    st.markdown("---")
    st.markdown("👈 Use the sidebar to navigate between projects.")


# ============================================================
#  PROJECT 1 PAGE
# ============================================================

elif page == "📈 Project 1 – Time Series":
    st.title("📈 Project 1: Time Series Forecasting")
    st.markdown("**Dataset:** Monthly Airline Passengers (1949–1960) | **Goal:** Forecast 1961 passenger counts")
    st.markdown("---")

    # Load & process data
    with st.spinner("Loading dataset..."):
        df_raw = load_airline_data()
        df = compute_moving_averages(df_raw)
        sma_forecast, ema_forecast, drift = compute_forecast(df)

    # ── Dataset Preview ──────────────────────────────────────
    st.header("1. Dataset Preview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_raw.head(12), use_container_width=True)
    with col2:
        st.subheader("Summary")
        st.metric("Total Records", len(df_raw))
        st.metric("Min Passengers", f"{df_raw['Passengers'].min()}k")
        st.metric("Max Passengers", f"{df_raw['Passengers'].max()}k")
        st.metric("Mean Passengers", f"{df_raw['Passengers'].mean():.1f}k")

    st.markdown("---")

    # ── EDA Plots ────────────────────────────────────────────
    st.header("2. Exploratory Data Analysis")
    st.pyplot(plot_time_series(df))

    st.markdown("---")

    # ── SMA ──────────────────────────────────────────────────
    st.header("3. Simple Moving Average (SMA)")
    st.markdown("""
    SMA smooths the data by averaging a fixed number of past observations.  
    A **larger window** = smoother line, but slower to react to changes.
    """)
    st.pyplot(plot_sma(df))

    st.markdown("---")

    # ── EMA ──────────────────────────────────────────────────
    st.header("4. Exponential Moving Average (EMA)")
    st.markdown("""
    EMA gives **more weight to recent values**, making it more responsive than SMA.  
    Useful when recent trends matter more.
    """)
    st.pyplot(plot_ema(df))

    st.markdown("---")

    # ── SMA vs EMA ───────────────────────────────────────────
    st.header("5. SMA vs EMA Comparison")
    st.pyplot(plot_sma_vs_ema(df))

    st.markdown("---")

    # ── MAE ──────────────────────────────────────────────────
    st.header("6. Error Evaluation (MAE)")
    actual = df["Passengers"]
    mae_results = {
        "SMA-3" : mae(actual, df["SMA_3"]),
        "SMA-6" : mae(actual, df["SMA_6"]),
        "SMA-12": mae(actual, df["SMA_12"]),
        "EMA-3" : mae(actual, df["EMA_3"]),
        "EMA-6" : mae(actual, df["EMA_6"]),
        "EMA-12": mae(actual, df["EMA_12"]),
    }
    best_method = min(mae_results, key=mae_results.get)

    cols = st.columns(6)
    for col, (method, error) in zip(cols, mae_results.items()):
        label = f"⭐ {method}" if method == best_method else method
        col.metric(label, f"{error:.2f}k")

    st.pyplot(plot_mae(mae_results))
    st.success(f"✅ Best Method: **{best_method}** with MAE = {mae_results[best_method]:.2f}k passengers")

    st.markdown("---")

    # ── Forecast ─────────────────────────────────────────────
    st.header("7. Forecast for 1961")
    st.markdown(f"""
    Forecast method: **Seasonal repeat of last 12 months + linear trend drift of {drift:+.1f}k passengers/year**
    """)

    col1, col2 = st.columns(2)
    col1.metric("SMA-12 Forecast Range", f"{sma_forecast.min():.0f}k – {sma_forecast.max():.0f}k")
    col2.metric("EMA-12 Forecast Range", f"{ema_forecast.min():.0f}k – {ema_forecast.max():.0f}k")

    st.pyplot(plot_forecast(df, sma_forecast, ema_forecast))

    # Forecast table
    forecast_df = pd.DataFrame({
        "Month": sma_forecast.index.strftime("%b %Y"),
        "SMA-12 Forecast (k)": sma_forecast.round(1).values,
        "EMA-12 Forecast (k)": ema_forecast.round(1).values,
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# ============================================================
#  PROJECT 2 PAGE
# ============================================================

elif page == "🤖 Project 2 – Diabetes Classification":
    st.title("🤖 Project 2: Diabetes Classification")
    st.markdown("**Dataset:** Pima Indians Diabetes | **Model:** Random Forest Classifier")
    st.markdown("---")

    # Load & train
    with st.spinner("Loading data and training model..."):
        df_raw = load_diabetes_data()
        df, X, y, X_train, X_test, acc, report, cm, importances = preprocess_and_train(df_raw)

    # ── Dataset Preview ──────────────────────────────────────
    st.header("1. Dataset Preview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col2:
        st.subheader("Summary")
        st.metric("Total Records", df_raw.shape[0])
        st.metric("Features", df_raw.shape[1] - 1)
        st.metric("Diabetic Cases", int(df_raw["Outcome"].sum()))
        st.metric("Non-Diabetic Cases", int((df_raw["Outcome"] == 0).sum()))

    st.markdown("---")

    # ── Preprocessing ────────────────────────────────────────
    st.header("2. Data Preprocessing")
    st.markdown("""
    Some columns use `0` as a placeholder for missing values (e.g. `Glucose=0` or `BMI=0` are biologically impossible).  
    These zeros are replaced with `NaN` and then filled with the **column median**.
    
    **Columns treated:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
    """)

    st.markdown("---")

    # ── Train/Test Split ─────────────────────────────────────
    st.header("3. Train / Test Split")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", X_train.shape[0])
    col2.metric("Testing Samples",  X_test.shape[0])
    col3.metric("Test Size", "20%")

    st.markdown("---")

    # ── Model Performance ────────────────────────────────────
    st.header("4. Model Performance")
    st.metric("🎯 Model Accuracy", f"{acc * 100:.2f}%",
              help="Percentage of correct predictions on the test set")

    st.markdown("#### Classification Report")
    st.code(report, language="text")

    st.markdown("---")

    # ── Visualizations ───────────────────────────────────────
    st.header("5. Visualizations")

    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])

    with col1:
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(cm))

    with col2:
        st.subheader("Feature Importances")
        st.pyplot(plot_feature_importance(importances))

    with col3:
        st.subheader("Class Distribution")
        st.pyplot(plot_class_distribution(y))

    st.markdown("---")
    st.markdown("""
    #### 📝 Key Takeaways
    - **Glucose** is the most important feature for predicting diabetes.
    - The dataset is slightly **imbalanced** (~65% non-diabetic, ~35% diabetic).
    - Random Forest achieves solid accuracy without any hyperparameter tuning.
    """)