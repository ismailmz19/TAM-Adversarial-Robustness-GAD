import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import linregress

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

COLORS = {'Amazon':'#2196F3','Facebook':'#F44336','Reddit':'#4CAF50','BlogCatalog':'#FF9800'}
DATASETS = ['Amazon','Facebook','Reddit','BlogCatalog']
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

def load_v2(d): return np.load(f'results_camouflage_v2_{d}.npy', allow_pickle=True).item()
def load_v1(d): return np.load(f'results_camouflage_{d}.npy', allow_pickle=True).item()

data = {d: load_v2(d) for d in DATASETS}
v1   = {d: load_v1(d) for d in DATASETS}

# ── Figure 1: Attack effectiveness ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
k_vals = [0, 1, 2, 3, 5]

for dataset in DATASETS:
    d1, d2 = v1[dataset], data[dataset]
    aurocs = [d2['results'][0][0.0]['auroc']]
    auprc  = [d2['results'][0][0.0]['auprc']]
    for k in [1, 2, 3, 5]:
        if k in d1['results']:
            aurocs.append(d1['results'][k]['auroc_base'])
            auprc.append(d1['results'][k]['auprc_base'])
        else:
            aurocs.append(d2['results'][k][0.0]['auroc'])
            auprc.append(d2['results'][k][0.0]['auprc'])
    c = COLORS[dataset]
    axes[0].plot(k_vals, aurocs, marker='o', color=c, linewidth=2, markersize=6, label=dataset)
    axes[1].plot(k_vals, auprc,  marker='s', color=c, linewidth=2, markersize=6, label=dataset, linestyle='--')

for ax, ylabel, title in zip(axes, ['AUROC','AUPRC'],
    ['AUROC under Relation Camouflage Attack','AUPRC under Relation Camouflage Attack']):
    ax.set_xlabel('Attack Intensity (k edges per anomalous node)')
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(k_vals); ax.legend(); ax.set_xlim(-0.2, 5.2)
axes[0].axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
fig.suptitle('Figure 1: Effect of Relation Camouflage Attack on TAM', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f'{OUT}/fig1_attack_effectiveness.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig1_attack_effectiveness.png', bbox_inches='tight')
plt.close(); print("Figure 1 saved.")

# ── Figure 2: Defense recovery ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(DATASETS)); width = 0.25

for ax, mk, title, yl in zip(axes, ['auroc','auprc'],
    ['AUROC: Clean vs Attacked vs Defended (k=5, τ=0.1)',
     'AUPRC: Clean vs Attacked vs Defended (k=5, τ=0.1)'], ['AUROC','AUPRC']):
    clean = [data[d]['results'][0][0.0][mk] for d in DATASETS]
    att   = [data[d]['results'][5][0.0][mk] for d in DATASETS]
    defs  = [data[d]['results'][5][0.1][mk] for d in DATASETS]
    ax.bar(x-width, clean, width, label='Clean (k=0)',        color='#90CAF9', edgecolor='white')
    ax.bar(x,       att,   width, label='Attacked (k=5)',      color='#EF9A9A', edgecolor='white')
    ax.bar(x+width, defs,  width, label='Defended (k=5,τ=0.1)',color='#A5D6A7', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(DATASETS)
    ax.set_ylabel(yl); ax.set_title(title, fontsize=11); ax.legend()
    if mk == 'auroc':
        for i,(a,df) in enumerate(zip(att,defs)):
            rec = df-a; col = '#2E7D32' if rec>0 else '#C62828'
            ax.annotate(f'{rec:+.3f}', xy=(x[i]+width/2, max(a,df)+0.01),
                        ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
fig.suptitle('Figure 2: Jaccard Defense Recovery at τ=0.1', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f'{OUT}/fig2_defense_recovery.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig2_defense_recovery.png', bbox_inches='tight')
plt.close(); print("Figure 2 saved.")

# ── Figure 3: Clustering vs attack drop ───────────────────────────────────────
clustering   = {'Amazon':0.6913,'Facebook':0.5268,'Reddit':0.0000,'BlogCatalog':0.1229}
attack_drop  = {d: data[d]['results'][0][0.0]['auroc'] - data[d]['results'][5][0.0]['auroc'] for d in DATASETS}

fig, ax = plt.subplots(figsize=(6, 5))
cc_arr   = np.array([clustering[d] for d in DATASETS])
drop_arr = np.array([attack_drop[d] for d in DATASETS])
slope, intercept, r, p, _ = linregress(cc_arr, drop_arr)

for d in DATASETS:
    ax.scatter(clustering[d], attack_drop[d], color=COLORS[d], s=150, zorder=5, edgecolors='white', linewidth=1.5)
    off = {'Amazon':(0.02,0.005),'Facebook':(0.02,0.005),'Reddit':(0.02,-0.015),'BlogCatalog':(0.02,0.008)}
    ax.annotate(d, (clustering[d]+off[d][0], attack_drop[d]+off[d][1]), fontsize=10, color=COLORS[d], fontweight='bold')

x_line = np.linspace(-0.05, 0.75, 100)
ax.plot(x_line, slope*x_line+intercept, '--', color='gray', linewidth=1.5, alpha=0.7, label=f'Trend (r={r:.2f}, p={p:.2f})')
ax.set_xlabel('Average Clustering Coefficient')
ax.set_ylabel('AUROC Drop (k=0 → k=5)')
ax.set_title('Figure 3: Clustering Coefficient vs. Attack Effectiveness')
ax.legend(); ax.set_xlim(-0.05, 0.80); ax.set_ylim(-0.02, 0.42)
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_clustering_vs_attack.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig3_clustering_vs_attack.png', bbox_inches='tight')
plt.close(); print("Figure 3 saved.")

# ── Figure 4: Jaccard aggressiveness ─────────────────────────────────────────
total   = {'Amazon':175608,'Facebook':27552,'Reddit':89500,'BlogCatalog':172783}
removed = {'Amazon':22749, 'Facebook':2175, 'Reddit':78516,'BlogCatalog':155942}
ratios  = {d: removed[d]/total[d]*100 for d in DATASETS}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = [COLORS[d] for d in DATASETS]

bars = axes[0].bar(DATASETS, [ratios[d] for d in DATASETS], color=colors, edgecolor='white', width=0.5)
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
axes[0].set_ylabel('Edges Removed (%)'); axes[0].set_title('Edge Removal Ratio (τ=0.1)')
axes[0].set_ylim(0, 105); axes[0].legend()

kept = [total[d]-removed[d] for d in DATASETS]
rem  = [removed[d] for d in DATASETS]
axes[1].bar(DATASETS, kept, color='#90CAF9', label='Kept',    width=0.5)
axes[1].bar(DATASETS, rem,  bottom=kept, color='#EF9A9A', label='Removed', width=0.5)
for i,d in enumerate(DATASETS):
    axes[1].text(i, total[d]+2000, f'{total[d]:,}', ha='center', fontsize=9)
axes[1].set_ylabel('Number of Edges'); axes[1].set_title('Absolute Edges: Kept vs. Removed (τ=0.1)'); axes[1].legend()

fig.suptitle('Figure 4: Jaccard Defense Aggressiveness Across Datasets', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_jaccard_aggressiveness.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig4_jaccard_aggressiveness.png', bbox_inches='tight')
plt.close(); print("Figure 4 saved.")

print(f"\nDone. All figures in ./{OUT}/")
