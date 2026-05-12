import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================================
#  Project 2: Predictive Model for Diabetes Classification
#  Dataset: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
#  Model: Random Forest Classifier
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── STEP 2: Load the Dataset ─────────────────────────────────
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("=" * 55)
print("STEP 2 ▸ Dataset loaded successfully")
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("=" * 55)
print(df.head())

# ── STEP 3: Explore the Data ─────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 ▸ Basic exploration")
print("=" * 55)
print("\n▸ Column info:")
print(df.info())

print("\n▸ Statistical summary:")
print(df.describe())

print("\n▸ Missing values:")
print(df.isnull().sum())

print("\n▸ Target distribution (Outcome):")
print(df["Outcome"].value_counts())
print("  0 = Not Diabetic  |  1 = Diabetic")

# ── STEP 4: Data Preprocessing ───────────────────────────────
# Some columns use 0 as a placeholder for missing data
# (e.g. Glucose=0, BMI=0 are biologically impossible).
# Replace zeros with NaN in those columns, then fill with median.

cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

print("\n" + "=" * 55)
print("STEP 4 ▸ Preprocessing done (zeros → median replacement)")
print("=" * 55)

# ── STEP 5: Feature Selection ────────────────────────────────
X = df.drop("Outcome", axis=1)   # all columns except target
y = df["Outcome"]                 # target variable

print("\n" + "=" * 55)
print("STEP 5 ▸ Features & target selected")
print(f"  Features : {list(X.columns)}")
print(f"  Target   : Outcome (0 or 1)")
print("=" * 55)

# ── STEP 6: Train / Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSTEP 6 ▸ Split complete")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ── STEP 7: Feature Scaling ──────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nSTEP 7 ▸ Features scaled with StandardScaler")

# ── STEP 8: Train the Model ──────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 55)
print("STEP 8 ▸ RandomForestClassifier trained (100 trees)")
print("=" * 55)

# ── STEP 9: Evaluate Performance ────────────────────────────
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\nSTEP 9 ▸ Model Evaluation")
print(f"  Accuracy : {acc * 100:.2f}%")
print("\n▸ Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))

# ── STEP 10: Visualizations ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 2 – Diabetes Prediction (Random Forest)", fontsize=14, fontweight="bold")

# --- Plot A: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Diabetic", "Diabetic"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# --- Plot B: Feature Importance ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Importance Score")

# --- Plot C: Target Distribution ---
counts = y.value_counts()
axes[2].pie(
    counts,
    labels=["Not Diabetic", "Diabetic"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90,
)
axes[2].set_title("Dataset Class Distribution")

plt.tight_layout()
plt.savefig("project2_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as  project2_results.png")
print("\n Project 2 complete!")