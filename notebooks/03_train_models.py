import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')

# Load feature dataset
df = pd.read_csv('../data/train_features.csv')

# Feature columns
feature_cols = [
    'word_count',
    'is_feeling_words',
    'sentiment_polarity',
    'positive_word_ratio',
    'negative_word_ratio',
    'emotion_word_count',
    'avg_word_length',
    'first_person_count',
    'intensifier_count',
    'exclamation_count',
    'hour_of_day',
    'day_of_week',
    'days_since_first'
]

# Input and targets
X = df[feature_cols].fillna(0)

y_val = df['valence']
y_aro = df['arousal']

# Train-validation split
X_tr, X_vl, yv_tr, yv_vl = train_test_split(
    X,
    y_val,
    test_size=0.2,
    random_state=42
)

_, _, ya_tr, ya_vl = train_test_split(
    X,
    y_aro,
    test_size=0.2,
    random_state=42
)

print("Training set size:", X_tr.shape[0], "rows")
print("Validation set size:", X_vl.shape[0], "rows\n")

# ==================================================
# RANDOM FOREST
# ==================================================

print("Training Random Forest...")

rf_val = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_val.fit(X_tr, yv_tr)

rf_aro = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_aro.fit(X_tr, ya_tr)

rf_pred_val = rf_val.predict(X_vl)
rf_pred_aro = rf_aro.predict(X_vl)

rf_r_val, _ = pearsonr(yv_vl, rf_pred_val)
rf_r_aro, _ = pearsonr(ya_vl, rf_pred_aro)

rf_rmse_val = np.sqrt(
    mean_squared_error(yv_vl, rf_pred_val)
)

rf_rmse_aro = np.sqrt(
    mean_squared_error(ya_vl, rf_pred_aro)
)

print("✅ RANDOM FOREST RESULTS:")
print(f"Valence → Pearson r: {rf_r_val:.3f} | RMSE: {rf_rmse_val:.3f}")
print(f"Arousal → Pearson r: {rf_r_aro:.3f} | RMSE: {rf_rmse_aro:.3f}\n")

# ==================================================
# XGBOOST
# ==================================================

print("Training XGBoost...")

xgb_val = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_val.fit(X_tr, yv_tr)

xgb_aro = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_aro.fit(X_tr, ya_tr)

xgb_pred_val = xgb_val.predict(X_vl)
xgb_pred_aro = xgb_aro.predict(X_vl)

xgb_r_val, _ = pearsonr(yv_vl, xgb_pred_val)
xgb_r_aro, _ = pearsonr(ya_vl, xgb_pred_aro)

xgb_rmse_val = np.sqrt(
    mean_squared_error(yv_vl, xgb_pred_val)
)

xgb_rmse_aro = np.sqrt(
    mean_squared_error(ya_vl, xgb_pred_aro)
)

print("✅ XGBOOST RESULTS:")
print(f"Valence → Pearson r: {xgb_r_val:.3f} | RMSE: {xgb_rmse_val:.3f}")
print(f"Arousal → Pearson r: {xgb_r_aro:.3f} | RMSE: {xgb_rmse_aro:.3f}\n")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

plt.figure(figsize=(10, 6))

importances = rf_val.feature_importances_

sorted_idx = np.argsort(importances)

plt.barh(
    [feature_cols[i] for i in sorted_idx],
    importances[sorted_idx]
)

plt.title('Feature Importance - Random Forest')

plt.tight_layout()

plt.savefig('../results/feature_importance.png')

plt.show()

# ==================================================
# FINAL TABLE
# ==================================================

print("=" * 50)
print("FINAL COMPARISON TABLE")
print("=" * 50)

print(f"{'Model':<20} {'Val r':>8} {'Aro r':>8} {'Val RMSE':>10}")

print("-" * 50)

print(f"{'Random Forest':<20} {rf_r_val:>8.3f} {rf_r_aro:>8.3f} {rf_rmse_val:>10.3f}")

print(f"{'XGBoost':<20} {xgb_r_val:>8.3f} {xgb_r_aro:>8.3f} {xgb_rmse_val:>10.3f}")

print("=" * 50)

print("\n✅ Baseline training complete!")
print("Feature importance chart saved.")
np.save(r'D:\MLproject semeval\results\rf_val_preds.npy', rf_pred_val)

np.save(r'D:\MLproject semeval\results\xgb_val_preds.npy', xgb_pred_val)

np.save(r'D:\MLproject semeval\results\true_val.npy', np.array(yv_vl))

print("Predictions saved!")