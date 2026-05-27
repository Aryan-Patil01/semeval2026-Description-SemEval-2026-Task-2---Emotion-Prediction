import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

# ── LOAD SAVED PREDICTIONS ───────────────────

rf_preds = np.load(r'D:\MLproject semeval\results\rf_val_preds.npy')

xgb_preds = np.load(r'D:\MLproject semeval\results\xgb_val_preds.npy')

true_vals = np.load(r'D:\MLproject semeval\results\true_val.npy')

# ── EVALUATION FUNCTION ──────────────────────

def evaluate(y_true, y_pred, name):

    r, _ = pearsonr(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)

    print(f"{name:<20} | Pearson r: {r:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

    return {
        'model': name,
        'pearson_r': r,
        'rmse': rmse,
        'mae': mae
    }

print("=" * 65)

print(f"{'MODEL':<20} | {'Pearson r':>10} | {'RMSE':>8} | {'MAE':>8}")

print("=" * 65)

results = []

results.append(evaluate(true_vals, rf_preds, "Random Forest"))

results.append(evaluate(true_vals, xgb_preds, "XGBoost"))

# ── SAVE RESULTS TABLE ──────────────────────

df_results = pd.DataFrame(results)

df_results.to_csv(
    r'D:\MLproject semeval\results\model_comparison.csv',
    index=False
)

print("\nResults saved to model_comparison.csv")

# ── ERROR ANALYSIS CHARTS ───────────────────

errors = true_vals - rf_preds

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Predicted vs true
axes[0].scatter(
    true_vals,
    rf_preds,
    alpha=0.4,
    color='steelblue',
    s=15
)

axes[0].plot([0, 4], [0, 4], 'r--')

axes[0].set_xlabel('True Valence')

axes[0].set_ylabel('Predicted Valence')

axes[0].set_title('Predicted vs True (RF)')

# Error histogram
axes[1].hist(
    errors,
    bins=30,
    color='coral',
    edgecolor='white'
)

axes[1].axvline(0, color='black', linestyle='--')

axes[1].set_xlabel('Prediction Error')

axes[1].set_title('Error Distribution')

# Model comparison
axes[2].bar(
    ['Random Forest', 'XGBoost'],
    [results[0]['pearson_r'], results[1]['pearson_r']],
    color=['steelblue', 'coral']
)

axes[2].set_ylim(0, 1)

axes[2].set_ylabel('Pearson r')

axes[2].set_title('Model Comparison')

plt.tight_layout()

plt.savefig(
    r'D:\MLproject semeval\results\evaluation_charts.png',
    dpi=150
)

plt.show()

print("Charts saved!")

# ── CROSS VALIDATION ─────────────────────────

print("\nRunning 5-Fold Cross Validation...\n")

df = pd.read_csv(
    r'D:\MLproject semeval\data\train_features.csv'
)

feature_cols = [
    'word_count',
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

X = df[feature_cols].fillna(0).values

y = df['valence'].values

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):

    X_tr, X_vl = X[tr_idx], X[vl_idx]

    y_tr, y_vl = y[tr_idx], y[vl_idx]

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr, y_tr)

    pred = rf.predict(X_vl)

    r, _ = pearsonr(y_vl, pred)

    scores.append(r)

    print(f"Fold {fold+1}: Pearson r = {r:.3f}")

print(f"\n5-Fold CV Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

print("\nUse this number in your paper!") 