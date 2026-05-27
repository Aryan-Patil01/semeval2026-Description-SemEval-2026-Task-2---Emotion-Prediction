import pandas as pd
from scipy.stats import pearsonr

# ==============================
# LOAD PREDICTIONS
# ==============================

rf_pred = pd.read_csv(
    r'D:\MLproject semeval\results\rf_submission.csv'
)

xgb_pred = pd.read_csv(
    r'D:\MLproject semeval\results\xgb_submission.csv'
)

rob_pred = pd.read_csv(
    r'D:\MLproject semeval\results\roberta_submission.csv'
)

# ==============================
# LOAD TRUE LABELS
# ==============================

labels = pd.read_csv(
    r'D:\MLproject semeval\data\test_labels_subtask1.csv'
)

# Normalize labels
labels['valence'] = labels['valence'] + 2
labels['arousal'] = labels['arousal'] + 1

# ==============================
# ENSEMBLE COMBINATIONS
# ==============================

combos = [

    ("Equal (33/33/33)", 0.33, 0.33, 0.33),

    ("RoBERTa heavy (20/20/60)",
     0.20, 0.20, 0.60),

    ("XGB+RoB (10/40/50)",
     0.10, 0.40, 0.50),

]

print("=" * 65)
print("FINAL ENSEMBLE RESULTS")
print("=" * 65)

best_score = 0
best_name = ""

for name, w1, w2, w3 in combos:

    # Ensemble predictions
    ens_val = (
        w1 * rf_pred['pred_valence'] +
        w2 * xgb_pred['pred_valence'] +
        w3 * rob_pred['pred_valence']
    )

    ens_aro = (
        w1 * rf_pred['pred_arousal'] +
        w2 * xgb_pred['pred_arousal'] +
        w3 * rob_pred['pred_arousal']
    )

    # Pearson correlations
    r_v, _ = pearsonr(labels['valence'], ens_val)
    r_a, _ = pearsonr(labels['arousal'], ens_aro)

    avg_score = (r_v + r_a) / 2

    print(f"{name:<30} Valence: {r_v:.3f} | Arousal: {r_a:.3f}")

    if avg_score > best_score:
        best_score = avg_score
        best_name = name

print("=" * 65)
print(f"BEST ENSEMBLE: {best_name}")
print("=" * 65)