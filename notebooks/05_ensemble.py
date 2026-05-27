import numpy as np
from scipy.stats import pearsonr

rf_preds = np.load(r'D:\MLproject semeval\results\rf_val_preds.npy')

xgb_preds = np.load(r'D:\MLproject semeval\results\xgb_val_preds.npy')

true_vals = np.load(r'D:\MLproject semeval\results\true_val.npy')

ensemble_preds = (0.5 * rf_preds) + (0.5 * xgb_preds)

r, _ = pearsonr(true_vals, ensemble_preds)

print(f"Ensemble Valence Pearson r: {r:.3f}")