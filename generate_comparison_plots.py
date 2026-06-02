import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure matplotlib formatting for a clean, professional academic look
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Model file mapping (active models only)
modelli_info = {
    "Linear Regression": "Predizioni_Regressione_Lineare.csv",
    "Random Forest": "Predizioni_Random_Forest.csv",
    "Gradient Boosting": "Predizioni_Gradient_Boosting_(Tuned).csv",
    "LightGBM": "Predizioni_LightGBM_(Tuned).csv",
    "CatBoost": "Predizioni_CatBoost.csv"
}

# 1. Load data and compute metrics
data = {}
metrics = []

for name, filename in modelli_info.items():
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Skipping model {name}.")
        continue
    
    df = pd.read_csv(filename)
    data[name] = df
    
    y_true = df["REALE_NEXT_PPG"]
    y_pred = df["PREDIZIONE"]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Global Accuracy % (100 - WMAPE)
    somma_errori = np.sum(np.abs(y_true - y_pred))
    somma_reali = np.sum(y_true)
    accuracy = 100 - ((somma_errori / somma_reali) * 100) if somma_reali > 0 else 0
    
    metrics.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy": accuracy
    })

df_metrics = pd.DataFrame(metrics)
print("=== MODEL METRICS ===")
print(df_metrics.to_string(index=False, formatters={
    'MAE': '{:.3f}'.format,
    'RMSE': '{:.3f}'.format,
    'R2': '{:.3%}'.format,
    'Accuracy': '{:.2f}%'.format
}))

# Ensure directory Papers exists
os.makedirs("Papers", exist_ok=True)

# ----------------- PLOT 1: Performance Metrics Comparison -----------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Model Performance Metrics Comparison", fontweight="bold", y=0.96)

# Colors for models
colors = ["#e67e22", "#9b59b6", "#3498db", "#2980b9", "#1abc9c"]

# MAE
axs[0, 0].bar(df_metrics["Model"], df_metrics["MAE"], color=colors, edgecolor='#7f8c8d', alpha=0.9, width=0.6)
axs[0, 0].set_title("Mean Absolute Error (MAE)")
axs[0, 0].set_ylabel("Points per Game (PPG)")
axs[0, 0].grid(axis='y')
axs[0, 0].set_xticks(range(len(df_metrics["Model"])))
axs[0, 0].set_xticklabels(df_metrics["Model"], rotation=15, ha="right")

# RMSE
axs[0, 1].bar(df_metrics["Model"], df_metrics["RMSE"], color=colors, edgecolor='#7f8c8d', alpha=0.9, width=0.6)
axs[0, 1].set_title("Root Mean Squared Error (RMSE)")
axs[0, 1].set_ylabel("Points per Game (PPG)")
axs[0, 1].grid(axis='y')
axs[0, 1].set_xticks(range(len(df_metrics["Model"])))
axs[0, 1].set_xticklabels(df_metrics["Model"], rotation=15, ha="right")

# R2 Score
axs[1, 0].bar(df_metrics["Model"], df_metrics["R2"] * 100, color=colors, edgecolor='#7f8c8d', alpha=0.9, width=0.6)
axs[1, 0].set_title("Coefficient of Determination ($R^2$)")
axs[1, 0].set_ylabel("R-squared (%)")
axs[1, 0].grid(axis='y')
axs[1, 0].set_xticks(range(len(df_metrics["Model"])))
axs[1, 0].set_xticklabels(df_metrics["Model"], rotation=15, ha="right")

# Accuracy
axs[1, 1].bar(df_metrics["Model"], df_metrics["Accuracy"], color=colors, edgecolor='#7f8c8d', alpha=0.9, width=0.6)
axs[1, 1].set_title("Global Accuracy (100 - WMAPE)")
axs[1, 1].set_ylabel("Accuracy (%)")
axs[1, 1].set_ylim(bottom=max(0, df_metrics["Accuracy"].min() - 5))  # zoom in a bit to see differences
axs[1, 1].grid(axis='y')
axs[1, 1].set_xticks(range(len(df_metrics["Model"])))
axs[1, 1].set_xticklabels(df_metrics["Model"], rotation=15, ha="right")

# Add value labels on top of the bars
for ax, col, is_pct in zip(axs.flat, ["MAE", "RMSE", "R2", "Accuracy"], [False, False, True, False]):
    for i, v in enumerate(df_metrics[col]):
        val = v * 100 if is_pct else v
        label = f"{val:.1f}%" if (is_pct or col == "Accuracy") else f"{val:.2f}"
        ax.text(i, v * 100 if is_pct else v, f" {label}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Papers/model_comparison_metrics.png", dpi=300)
plt.close()
print("-> Saved Papers/model_comparison_metrics.png")

# ----------------- PLOT 2: Actual vs Predicted Scatter with Regression Lines -----------------
fig, axs = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle("Actual vs. Predicted Values with Fitted Regression Lines", fontweight="bold", y=0.96)
axs = axs.ravel()

# Hide the 6th subplot since we only have 5 models
fig.delaxes(axs[5])

for idx, (name, df) in enumerate(data.items()):
    ax = axs[idx]
    y_true = df["REALE_NEXT_PPG"]
    y_pred = df["PREDIZIONE"]
    
    # Scatter plot of predictions
    ax.scatter(y_true, y_pred, alpha=0.4, color="#34495e", edgecolor='none', s=25, label="Players")
    
    # Perfect prediction line (y = x)
    lims = [0, 35]
    ax.plot(lims, lims, color='#95a5a6', linestyle='--', linewidth=1.5, label="Perfect Prediction (y = x)")
    
    # Fit regression line: Predizione = m * Reale + q
    m, q = np.polyfit(y_true, y_pred, 1)
    x_fit = np.linspace(0, 35, 100)
    y_fit = m * x_fit + q
    ax.plot(x_fit, y_fit, color='#e74c3c', linestyle='-', linewidth=2, label=f"Linear Fit (m={m:.2f})")
    
    # Metrics textbox
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    textstr = '\n'.join((
        f"MAE: {mae:.2f}",
        f"$R^2$: {r2:.3f}",
        f"y = {m:.2f}x + {q:.2f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_title(name, fontweight="bold")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.5)
    if idx in [0, 3]:
        ax.set_ylabel("Predicted NEXT_PPG", fontsize=11)
    # Since the 6th subplot is deleted, the 3rd subplot (idx=2) is now on the bottom of its column
    if idx in [2, 3, 4]:
        ax.set_xlabel("Actual NEXT_PPG", fontsize=11)
    if idx == 0:
        ax.legend(loc="lower right", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig("Papers/model_regression_fits.png", dpi=300)
plt.close()
print("-> Saved Papers/model_regression_fits.png")

# ----------------- PLOT 3: Regression Curves on PPG (Current vs Next season) -----------------
plt.figure(figsize=(10, 7.5))

# Use one of the datasets to plot the raw points (they all have same PPG and REALE_NEXT_PPG)
any_model_df = list(data.values())[0]
plt.scatter(any_model_df["PPG"], any_model_df["REALE_NEXT_PPG"], alpha=0.15, color="#7f8c8d", s=30, label="Actual Data (Players)")

# Plot actual regression curve (fit on raw data)
p_actual = np.polyfit(any_model_df["PPG"], any_model_df["REALE_NEXT_PPG"], 2)
x_curve = np.linspace(any_model_df["PPG"].min(), any_model_df["PPG"].max(), 200)
plt.plot(x_curve, np.polyval(p_actual, x_curve), color="black", linestyle="--", linewidth=2.5, label="Actual Trend (Degree 2)")

# Plot model regression curves (fit on predicted values vs current PPG)
curve_colors = {
    "Linear Regression": "#e67e22",
    "Random Forest": "#9b59b6",
    "Gradient Boosting": "#3498db",
    "LightGBM": "#2980b9",
    "CatBoost": "#1abc9c"
}

for name, df in data.items():
    p_model = np.polyfit(df["PPG"], df["PREDIZIONE"], 2)
    plt.plot(x_curve, np.polyval(p_model, x_curve), color=curve_colors[name], linewidth=2.0, label=f"{name} Curve")

plt.title("Regression Curves: Current PPG vs. Predicted NEXT_PPG", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Current Season Points Per Game (PPG)", fontsize=11)
plt.ylabel("Next Season Points Per Game (NEXT_PPG)", fontsize=11)
plt.xlim(any_model_df["PPG"].min() - 0.5, any_model_df["PPG"].max() + 0.5)
plt.ylim(-1, 35)
plt.grid(True, alpha=0.4)
plt.legend(loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.savefig("Papers/model_regression_curves.png", dpi=300)
plt.close()
print("-> Saved Papers/model_regression_curves.png")
print("All plots generated successfully!")
