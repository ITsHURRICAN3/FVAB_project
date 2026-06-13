import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load Cleaned Dataset
try:
    df = pd.read_csv("Dataset_Pulito_Pre_Addestramento.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Feature selection
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

# Split train and validation (using everything except 2023-24 for training)
df_train = df[df['SEASON'] != '2023-24']
X_train = df_train[feature_cols]
y_train = df_train['NEXT_PPG']

# 2. Define and train all 5 models
print("Training models...")
modelli = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=4, random_state=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=20, random_state=0, verbose=-1
    ),
    "CatBoost": CatBoostRegressor(random_state=0, verbose=0)
}

importances_dict = {}

for name, model in modelli.items():
    model.fit(X_train, y_train)
    if name == "Linear Regression":
        # Standardized coefficients (magnitude) as proxy for feature importance
        coefs = model.named_steps['linearregression'].coef_
        imp = np.abs(coefs)
        # Normalize to sum to 100% just for visualization scaling consistency
        imp = (imp / np.sum(imp)) * 100
    elif name == "LightGBM":
        # Get feature importance (gain split or gain, gain is preferred)
        imp = model.booster_.feature_importance(importance_type='gain')
        imp = (imp / np.sum(imp)) * 100
    else:
        imp = model.feature_importances_
        # Ensure it sums to 100
        imp = (imp / np.sum(imp)) * 100
        
    importances_dict[name] = imp
    print(f"Model {name} trained and feature importances extracted.")

# Create DataFrames for plotting
importances_dfs = {}
for name, imp in importances_dict.items():
    importances_dfs[name] = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': imp
    }).sort_values(by='Importance', ascending=True)

# 3. Plotting
# Configure matplotlib formatting for clean look
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--'
})

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Feature Importance Comparison across All Models", fontweight="bold", fontsize=16, y=0.97)
axs = axs.ravel()

# Remove the 6th axis since we only have 5 models
fig.delaxes(axs[5])

colors = {
    "Linear Regression": "#e67e22",
    "Random Forest": "#9b59b6",
    "Gradient Boosting": "#3498db",
    "LightGBM": "#2980b9",
    "CatBoost": "#1abc9c"
}

for idx, (name, df_imp) in enumerate(importances_dfs.items()):
    ax = axs[idx]
    
    # Sort top 15 features for readability in subplot, or plot all 25 if they fit
    # Let's plot all 25 but with a slightly wider spacing or top 15. Let's do top 15 to keep it clean.
    df_plot = df_imp.tail(15)
    
    ax.barh(df_plot["Feature"], df_plot["Importance"], color=colors[name], height=0.6, edgecolor='none')
    
    title_suffix = " (Std Coef %)" if name == "Linear Regression" else " (Relative %)"
    ax.set_title(name + title_suffix, fontweight="bold", fontsize=12)
    ax.set_xlabel("Relative Importance (%)", fontsize=10)
    ax.grid(axis='x', alpha=0.5)
    
    # Add values on the bar tips
    for i, v in enumerate(df_plot["Importance"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=8, color="#555555")
        
    ax.set_xlim(0, max(df_plot["Importance"].max() + 5, 10))

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
os.makedirs("Papers", exist_ok=True)
output_path = "Papers/all_models_feature_importance.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Comparison plot successfully saved to '{output_path}'")
