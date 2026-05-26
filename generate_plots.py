import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# Configure matplotlib formatting
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--'
})

# 1. Load Cleaned Dataset
df = pd.read_csv("Dataset_Pulito_Pre_Addestramento.csv")

# ----------------- PLOT 1: Aging Curve -----------------
plt.figure(figsize=(7, 4.5))
age_grouped = df.groupby("AGE")["PPG"].agg(["mean", "std", "count"]).reset_index()
# Calculate standard error
sem = age_grouped["std"] / np.sqrt(age_grouped["count"])

plt.plot(age_grouped["AGE"], age_grouped["mean"], marker="o", color="#d62728", linewidth=2.5, label="Mean PPG")
plt.fill_between(age_grouped["AGE"], age_grouped["mean"] - sem, age_grouped["mean"] + sem, color="#d62728", alpha=0.15, label="Standard Error")
plt.axvline(x=27, color="black", linestyle="--", alpha=0.7, label="Athletic Peak (27 yrs)")

plt.title("NBA Player Offensive Output (PPG) vs Age", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Age", fontsize=11)
plt.ylabel("Points Per Game (PPG)", fontsize=11)
plt.xlim(df["AGE"].min() - 0.5, df["AGE"].max() + 0.5)
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("Papers/aging_curve.png", dpi=300)
plt.close()
print("-> Saved Papers/aging_curve.png")

# ----------------- PLOT 2: Feature Importance -----------------
feature_cols = [
    "AGE", "HEIGHT", "MPG", "PPG", "GP", "W",
    "TRB", "AST", "STL", "BLK",
    "FG%", "TS%",
    "USG%", "NET_RATING",
    "OREB_PCT", "DREB_PCT", "AST_PCT",
    "WIN_RATE", "PEAK_AGE_DIST",
    "PREV_PPG", "PREV_PPG_2",
    "PREV_USG%", "PREV_GP",
    "PPG_MOMENTUM", "CAREER_SEASON_NUM"
]

# Split train and validation
df_train = df[df['SEASON'] != '2023-24']
X_train = df_train[feature_cols]
y_train = df_train['NEXT_PPG']

df_val = df[df['SEASON'] == '2023-24']
X_val = df_val[feature_cols]
y_val = df_val['NEXT_PPG']

# Train CatBoost
catboost_model = CatBoostRegressor(random_state=0, verbose=0)
catboost_model.fit(X_train, y_train)

# Get feature importance
importances = catboost_model.get_feature_importance()
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)  # Ascending for horizontal bar chart

plt.figure(figsize=(7, 6.5))
plt.barh(feat_imp["Feature"], feat_imp["Importance"], color="#1f77b4", height=0.7)
plt.title("CatBoost Feature Importance", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Relative Importance (%)", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Papers/feature_importance.png", dpi=300)
plt.close()
print("-> Saved Papers/feature_importance.png")

# ----------------- PLOT 3: Actual vs Predicted PPG -----------------
pred_val = catboost_model.predict(X_val)

plt.figure(figsize=(6, 6))
plt.scatter(y_val, pred_val, alpha=0.5, color="#2ca02c", edgecolor='none', s=45)

lims = [
    min(y_val.min(), pred_val.min()),
    max(y_val.max(), pred_val.max())
]
plt.plot(lims, lims, color='red', linestyle='--', linewidth=2, label="Perfect Prediction (y=x)")
plt.title("Actual vs. Predicted PPG (CatBoost, Season 2023-24)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Actual NEXT_PPG (2024-25)", fontsize=11)
plt.ylabel("Predicted NEXT_PPG (2024-25)", fontsize=11)
plt.xlim(lims[0]-1, lims[1]+1)
plt.ylim(lims[0]-1, lims[1]+1)
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Papers/actual_vs_predicted.png", dpi=300)
plt.close()
print("-> Saved Papers/actual_vs_predicted.png")
