import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Load predictions
pred = pd.read_csv(r'D:\MLproject semeval\results\submission.csv')

# Load true labels
labels = pd.read_csv(r'D:\MLproject semeval\data\test_labels_subtask1.csv')

# Merge correctly
merged = pred.merge(labels, on='text_id')

# Metrics
r_val, _ = pearsonr(merged['valence'], merged['pred_valence'])
r_aro, _ = pearsonr(merged['arousal'], merged['pred_arousal'])

rmse_val = np.sqrt(mean_squared_error(
    merged['valence'],
    merged['pred_valence']
))

rmse_aro = np.sqrt(mean_squared_error(
    merged['arousal'],
    merged['pred_arousal']
))

print("=" * 50)
print("🎯 REAL TEST SET RESULTS")
print("=" * 50)

print(f"Valence Pearson r : {r_val:.3f}")
print(f"Valence RMSE      : {rmse_val:.3f}\n")

print(f"Arousal Pearson r : {r_aro:.3f}")
print(f"Arousal RMSE      : {rmse_aro:.3f}")

print("=" * 50)
print(f"Rows evaluated: {len(merged)}")