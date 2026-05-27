import matplotlib.pyplot as plt
import numpy as np

# ── Final Model Scores ─────────────────────────────
models = [
    'Random\nForest',
    'XGBoost',
    'RoBERTa',
    'Ensemble\n(Best)'
]

# Real TEST scores
val_scores = [0.572, 0.572, 0.659, 0.669]
aro_scores = [0.224, 0.224, 0.495, 0.504]

x = np.arange(len(models))
width = 0.35

# ── Create Figure ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    'SemEval 2026 — Emotion Prediction Results\nTeam 13, KLE Tech',
    fontsize=13,
    fontweight='bold'
)

# ── VALENCE CHART ──────────────────────────────────
bars1 = axes[0].bar(
    x,
    val_scores,
    width * 1.5,
    color=['#4472C4','#ED7D31','#A9D18E','#1D9E75'],
    edgecolor='white',
    linewidth=1.5
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('Pearson r', fontsize=11)

axes[0].set_title(
    'Valence Prediction',
    fontsize=12,
    fontweight='bold'
)

axes[0].axhline(
    y=0.5,
    color='red',
    linestyle='--',
    alpha=0.5,
    label='r = 0.5 threshold'
)

axes[0].legend(fontsize=9)

for bar, val in zip(bars1, val_scores):

    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{val:.3f}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

# ── AROUSAL CHART ──────────────────────────────────
bars2 = axes[1].bar(
    x,
    aro_scores,
    width * 1.5,
    color=['#4472C4','#ED7D31','#A9D18E','#1D9E75'],
    edgecolor='white',
    linewidth=1.5
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=10)
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel('Pearson r', fontsize=11)

axes[1].set_title(
    'Arousal Prediction',
    fontsize=12,
    fontweight='bold'
)

axes[1].axhline(
    y=0.4,
    color='red',
    linestyle='--',
    alpha=0.5,
    label='r = 0.4 threshold'
)

axes[1].legend(fontsize=9)

for bar, val in zip(bars2, aro_scores):

    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{val:.3f}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

# ── Final Formatting ───────────────────────────────
plt.tight_layout()

plt.savefig(
    r'D:\MLproject semeval\results\final_results_dashboard.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\n✅ Final dashboard saved!")
print("Location:")
print(r'D:\MLproject semeval\results\final_results_dashboard.png')