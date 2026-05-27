import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────

labels = pd.read_csv(
    r'D:\MLproject semeval\data\test_labels_subtask1.csv'
)

labels['valence'] = labels['valence'] + 2
labels['arousal'] = labels['arousal'] + 1

# Load predictions
submission = pd.read_csv(
    r'D:\MLproject semeval\results\submission.csv'
)

# Merge predictions + labels
merged = submission.merge(labels, on='text_id')

true_val  = merged['valence']
pred_val  = merged['pred_valence']

true_aro  = merged['arousal']
pred_aro  = merged['pred_arousal']

errors_val = true_val - pred_val
errors_aro = true_aro - pred_aro

# ── Build dashboard ──────────────────────────────────

fig = plt.figure(figsize=(14, 10))

fig.suptitle(
    'Error Analysis — SemEval 2026 Emotion Prediction',
    fontsize=14,
    fontweight='bold',
    y=0.98
)

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.4,
    wspace=0.35
)

# ── Chart 1 ──────────────────────────────────────────
# Predicted vs True — Valence

ax1 = fig.add_subplot(gs[0, 0])

ax1.scatter(
    true_val,
    pred_val,
    alpha=0.3,
    s=12,
)

ax1.plot([0,4],[0,4], 'r--', linewidth=1.5)

r_v, _ = pearsonr(true_val, pred_val)

ax1.set_xlabel('True Valence')
ax1.set_ylabel('Predicted Valence')

ax1.set_title(
    f'Valence: Predicted vs True (r={r_v:.3f})'
)

# ── Chart 2 ──────────────────────────────────────────
# Predicted vs True — Arousal

ax2 = fig.add_subplot(gs[0, 1])

ax2.scatter(
    true_aro,
    pred_aro,
    alpha=0.3,
    s=12,
)

ax2.plot([0,2],[0,2], 'r--', linewidth=1.5)

r_a, _ = pearsonr(true_aro, pred_aro)

ax2.set_xlabel('True Arousal')
ax2.set_ylabel('Predicted Arousal')

ax2.set_title(
    f'Arousal: Predicted vs True (r={r_a:.3f})'
)

# ── Chart 3 ──────────────────────────────────────────
# Error distribution

ax3 = fig.add_subplot(gs[1, 0])

ax3.hist(
    errors_val,
    bins=40,
    edgecolor='white',
    alpha=0.8
)

ax3.axvline(
    0,
    color='red',
    linestyle='--',
    linewidth=1.5
)

ax3.axvline(
    errors_val.mean(),
    color='orange',
    linestyle='-',
    linewidth=1.5
)

ax3.set_xlabel('Prediction Error')

ax3.set_ylabel('Count')

ax3.set_title('Valence Error Distribution')

# ==============================
# CHART 4 — Essays vs Feeling Words
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

train_full = pd.read_csv(r'D:\MLproject semeval\data\train_features.csv')

X_tr2, X_vl2, yv_tr2, yv_vl2 = train_test_split(
    train_full[['word_count','is_feeling_words','sentiment_polarity',
                'positive_word_ratio','negative_word_ratio','emotion_word_count',
                'avg_word_length','first_person_count','intensifier_count',
                'exclamation_count','hour_of_day','day_of_week',
                'days_since_first']].fillna(0),
    train_full['valence'],
    test_size=0.2,
    random_state=42
)

rf2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf2.fit(X_tr2, yv_tr2)

val_preds2 = rf2.predict(X_vl2)

# Split validation set
val_df2 = train_full.iloc[X_vl2.index].copy()
val_df2['pred'] = val_preds2

essays_v = val_df2[val_df2['is_feeling_words'] == 0]
fwords_v = val_df2[val_df2['is_feeling_words'] == 1]

r_e = pearsonr(essays_v['valence'], essays_v['pred'])[0]
r_f = pearsonr(fwords_v['valence'], fwords_v['pred'])[0]

ax4 = fig.add_subplot(gs[1, 1])

bars = ax4.bar(
    ['Essays', 'Feeling Words'],
    [r_e, r_f],
    color=['steelblue', 'coral'],
    width=0.5,
    edgecolor='white'
)

ax4.set_ylim(0, 1)
ax4.set_ylabel('Pearson r (Valence)')
ax4.set_title('Performance: Essays vs Feeling Words')

for bar, val in zip(bars, [r_e, r_f]):
    ax4.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.02,
        f'{val:.3f}',
        ha='center',
        fontsize=11,
        fontweight='bold'
    )
# ── Save charts ──────────────────────────────────────

plt.savefig(
    r'D:\MLproject semeval\results\error_analysis.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\n✅ Error analysis charts saved!")

# ── Worst predictions ────────────────────────────────

merged['val_error'] = abs(
    merged['valence'] - merged['pred_valence']
)

worst = merged.nlargest(
    5,
    'val_error'
)[[
    'text_id',
    'valence',
    'pred_valence',
    'val_error'
]]

print("\n5 WORST PREDICTIONS:")
print(worst.to_string(index=False))