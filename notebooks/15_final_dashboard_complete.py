import matplotlib.pyplot as plt
import numpy as np

# ── REAL SCORES ─────────────────────────────────────
subtask1 = {'val': [0.572, 0.572, 0.659, 0.669],
            'aro': [0.224, 0.224, 0.495, 0.504]}

subtask2a = {'val': 0.299,
             'aro': 0.230}

subtask2b = {'val': 0.003,
             'aro': 0.243}

models_s1 = ['RF', 'XGB', 'RoBERTa', 'Ensemble']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SemEval 2026 — Complete Results',
             fontsize=13, fontweight='bold')
colors = ['#4472C4','#ED7D31','#A9D18E','#1D9E75']

# Chart 1: Subtask 1
x = np.arange(4)
b1 = axes[0].bar(x-0.2, subtask1['val'], 0.35, label='Valence',
                  color='steelblue', edgecolor='white')
b2 = axes[0].bar(x+0.2, subtask1['aro'], 0.35, label='Arousal',
                  color='coral', edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(models_s1, fontsize=9)
axes[0].set_ylim(0,1); axes[0].set_title('Subtask 1\n(Current emotion from text)', fontweight='bold')
axes[0].legend(fontsize=8); axes[0].set_ylabel('Pearson r')
for bar in [*b1, *b2]:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=7)

# Chart 2: Subtask 2a
axes[1].bar(['Valence\nchange','Arousal\nchange'],
             [subtask2a['val'], subtask2a['aro']],
             color=['steelblue','coral'], width=0.4, edgecolor='white')
axes[1].set_ylim(-0.3, 1); axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Subtask 2a\n(Predict emotion change)', fontweight='bold')
axes[1].set_ylabel('Pearson r')
for i, (lbl, val) in enumerate([('Val',subtask2a['val']),('Aro',subtask2a['aro'])]):
    axes[1].text(i, val+0.02, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

# Chart 3: Subtask 2b
axes[2].bar(['Valence\nforecast','Arousal\nforecast'],
             [subtask2b['val'], subtask2b['aro']],
             color=['#534AB7','#BA7517'], width=0.4, edgecolor='white')
axes[2].set_ylim(-0.3, 1); axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_title('Subtask 2b\n(Cross-wave forecasting)', fontweight='bold')
axes[2].set_ylabel('Pearson r')
for i, (lbl, val) in enumerate([('Val',subtask2b['val']),('Aro',subtask2b['aro'])]):
    axes[2].text(i, val+0.02, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(r'D:\MLproject semeval\results\complete_results_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Complete dashboard saved!")