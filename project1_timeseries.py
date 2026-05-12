# ── 1. Import Libraries ──────────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
})

# ── 2. Load Dataset ──────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df  = pd.read_csv(url, header=0, names=['Month', 'Passengers'])

df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
# ---------------- Data Preprocessing ----------------

# Check missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Ensure data is sorted by date
df = df.sort_index()

print("=" * 55)
print("  Airline Passengers Dataset – Summary")
print("=" * 55)
print(f"  Records : {len(df)}")
print(f"  Period  : {df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')}")
print(f"  Min     : {df['Passengers'].min()} thousand")
print(f"  Max     : {df['Passengers'].max()} thousand")
print(f"  Mean    : {df['Passengers'].mean():.1f} thousand")
print("=" * 55)

# ── 3. Compute Moving Averages ───────────────────────────────────────────────
# Simple Moving Average (SMA)
df['SMA_3']  = df['Passengers'].rolling(window=3).mean()
df['SMA_6']  = df['Passengers'].rolling(window=6).mean()
df['SMA_12'] = df['Passengers'].rolling(window=12).mean()

# Exponential Moving Average (EMA)
df['EMA_3']  = df['Passengers'].ewm(span=3,  adjust=False).mean()
df['EMA_6']  = df['Passengers'].ewm(span=6,  adjust=False).mean()
df['EMA_12'] = df['Passengers'].ewm(span=12, adjust=False).mean()

print("\nMoving averages computed (SMA & EMA – windows 3, 6, 12 months)")

# ── 4. MAE Error Measurement ─────────────────────────────────────────────────
def mae(actual, predicted):
    common = actual.index.intersection(predicted.dropna().index)
    return np.mean(np.abs(actual[common] - predicted[common]))

actual = df['Passengers']
mae_results = {
    'SMA-3' : mae(actual, df['SMA_3']),
    'SMA-6' : mae(actual, df['SMA_6']),
    'SMA-12': mae(actual, df['SMA_12']),
    'EMA-3' : mae(actual, df['EMA_3']),
    'EMA-6' : mae(actual, df['EMA_6']),
    'EMA-12': mae(actual, df['EMA_12']),
}

print("\nMAE Results:")
print("-" * 38)
best_method = min(mae_results, key=mae_results.get)
for method, error in mae_results.items():
    tag = " ← Best" if method == best_method else ""
    print(f"  {method:<8}: {error:6.2f} thousand{tag}")
print("-" * 38)

# ── 5. Forecast Future Values (1961) ─────────────────────────────────────────
# Seasonal repeat of last 12 months + linear trend correction
# This is better than a flat-line forecast because the data has a clear upward trend

x = np.arange(len(df))

# Calculate trend using linear regression
slope, *_ = linregress(x, df['Passengers'].values)

# Expected increase over the next 12 months
drift = slope * 12

# Use the last 12 months to preserve seasonality
last_12 = df['Passengers'].iloc[-12:].values

# Generate future monthly dates for 1961
future_dates = pd.date_range(start='1961-01', periods=12, freq='MS')

# Forecast using seasonal repeat + trend drift
sma_forecast = pd.Series(last_12 + drift, index=future_dates)

# EMA forecast using EMA-based historical pattern
ema_forecast = pd.Series(
    df['EMA_12'].iloc[-12:].values + drift,
    index=future_dates
)

print(f"\nForecast 1961 | seasonal repeat + trend drift of {drift:+.1f}k passengers")
print(f"  SMA-12 range: {sma_forecast.min():.1f}k – {sma_forecast.max():.1f}k")
print(f"  EMA-12 range: {ema_forecast.min():.1f}k – {ema_forecast.max():.1f}k")

# ── 6. Plots ──────────────────────────────────────────────────────────────────

# ── Plot 1: EDA – Time Series + Boxplot by Year ───────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

axes[0].plot(df.index, df['Passengers'], color='#2196F3', linewidth=2,
             label='Monthly Passengers')
axes[0].fill_between(df.index, df['Passengers'], alpha=0.15, color='#2196F3')
axes[0].set_title('Monthly Airline Passengers (1949–1960)')
axes[0].set_ylabel('Passengers (thousands)')
axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[0].legend()

# Boxplot using actual years in the dataset
df['Year'] = df.index.year
years  = sorted(df['Year'].unique())
yearly = [df[df['Year'] == y]['Passengers'].values for y in years]
bp = axes[1].boxplot(yearly, labels=years, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#90CAF9')
axes[1].set_title('Passenger Distribution by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Passengers (thousands)')
df.drop(columns=['Year'], inplace=True)

plt.tight_layout()
plt.savefig('plot1_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1_eda.png")

# ── Plot 2: SMA ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Passengers'], color='#90A4AE', linewidth=1.5, alpha=0.8, label='Actual')
ax.plot(df.index, df['SMA_3'],  color='#F44336', linewidth=2, linestyle='--', label='SMA (window=3)')
ax.plot(df.index, df['SMA_6'],  color='#FF9800', linewidth=2, linestyle='--', label='SMA (window=6)')
ax.plot(df.index, df['SMA_12'], color='#4CAF50', linewidth=2.5, label='SMA (window=12)')
ax.set_title('Simple Moving Average (SMA) – Airline Passengers')
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.tight_layout()
plt.savefig('plot2_sma.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot2_sma.png")

# ── Plot 3: EMA ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Passengers'], color='#90A4AE', linewidth=1.5, alpha=0.8, label='Actual')
ax.plot(df.index, df['EMA_3'],  color='#E91E63', linewidth=2, linestyle='--', label='EMA (span=3)')
ax.plot(df.index, df['EMA_6'],  color='#9C27B0', linewidth=2, linestyle='--', label='EMA (span=6)')
ax.plot(df.index, df['EMA_12'], color='#3F51B5', linewidth=2.5, label='EMA (span=12)')
ax.set_title('Exponential Moving Average (EMA) – Airline Passengers')
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.tight_layout()
plt.savefig('plot3_ema.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot3_ema.png")

# ── Plot 4: SMA vs EMA Comparison ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Passengers'], color='#B0BEC5', linewidth=1.5, alpha=0.9, label='Actual')
ax.plot(df.index, df['SMA_12'], color='#4CAF50', linewidth=2.5, label='SMA (window=12)')
ax.plot(df.index, df['EMA_12'], color='#3F51B5', linewidth=2.5, linestyle='--', label='EMA (span=12)')
ax.set_title('SMA vs EMA Comparison (window = 12 months)')
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.tight_layout()
plt.savefig('plot4_sma_vs_ema.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot4_sma_vs_ema.png")

# ── Plot 5: Forecast 1961 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df.index, df['Passengers'], color='#607D8B', linewidth=1.5, alpha=0.8,
        label='Historical Passengers')
ax.plot(df.index, df['SMA_12'], color='#4CAF50', linewidth=2,
        label='SMA-12 (historical)')
ax.plot(df.index, df['EMA_12'], color='#3F51B5', linewidth=2, linestyle='--',
        label='EMA-12 (historical)')
ax.plot(future_dates, sma_forecast, color='#4CAF50', linewidth=2.5,
        linestyle=':', marker='o', markersize=5, label='SMA-12 Forecast (1961)')
ax.plot(future_dates, ema_forecast, color='#3F51B5', linewidth=2.5,
        linestyle=':', marker='s', markersize=5, label='EMA-12 Forecast (1961)')
ax.axvline(pd.Timestamp('1961-01'), color='red', linestyle='--', alpha=0.6,
           label='Forecast Start')
ax.set_title('Airline Passengers – Forecast for 1961 (SMA vs EMA)')
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.tight_layout()
plt.savefig('plot5_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot5_forecast.png")

# ── Plot 6: MAE Bar Chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors  = ['#EF9A9A', '#EF5350', '#C62828', '#90CAF9', '#1E88E5', '#0D47A1']
methods = list(mae_results.keys())
errors  = list(mae_results.values())
bars = ax.bar(methods, errors, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
ax.set_title('MAE Comparison – SMA vs EMA')
ax.set_ylabel('Mean Absolute Error (thousands)')
ax.set_ylim(0, max(errors) * 1.2)
plt.tight_layout()
plt.savefig('plot6_mae.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot6_mae.png")

print("\n All done! 6 plots saved.")