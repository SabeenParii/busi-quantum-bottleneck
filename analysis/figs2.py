"""Result figures of the manuscript, drawn at the printed size (6.3 in wide, 9 pt text) and fully vector.
Usage: RESULTS=<folder with the analysis CSVs> OUT=<figure folder> python figs2.py
"""
import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.colors as mc

R = os.environ.get('RESULTS', 'results/'); OUT = os.environ.get('OUT', 'figures')
W = 6.3
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7})
C = pd.read_csv(R + 'contrasts.csv'); L = pd.read_csv(R + 'model_levels.csv')
DS = [('busi', 'BUSI'), ('busbra', 'BUS-BRA'), ('breast', 'BrEaST')]
NAME = {'q4': 'Q', 'qp4': 'Qp', 's4': 'S', 'tr4': 'Tr', 'cl4': 'Cl', 'id4': 'Id', 'full': 'Full', 'unet': 'U-Net'}
lab = lambda c: ' vs '.join(NAME[x.strip()] for x in c.split('-'))
BLUE, ORANGE = '#1f4e79', '#b4541f'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf')); fig.savefig(os.path.join(OUT, name + '.png'), dpi=300); plt.close(fig)


# ---------------------------------------------------------------- contrasts (forest plot)
rows = [('F1', 'q4 - qp4'), ('F1', 'q4 - s4'), ('F1', 'q4 - tr4'), ('F1', 'q4 - cl4'),
        ('F2', 'tr4 - full'), ('F2', 'q4 - full'), ('F2', 'tr4 - unet'), ('F2', 'q4 - unet')]
fig, axs = plt.subplots(1, 3, figsize=(W, 3.2), sharey=True)
for k, (ax, (ds, title)) in enumerate(zip(axs, DS)):
    fam = lambda f: f if ds == 'busi' else f'{f}_{ds}'
    ax.axvspan(-0.02, 0.02, color='#dbe7f3', lw=0); ax.axvline(0, color='#444', lw=0.7)
    for i, (f, c) in enumerate(rows):
        r = C[(C.family == fam(f)) & (C.contrast == c)].iloc[0]; y = len(rows) - 1 - i
        col = BLUE if f == 'F1' else ORANGE
        ax.plot([r.ci95_lo, r.ci95_hi], [y, y], color=col, lw=1.4)
        ax.plot(r['diff'], y, 'o', ms=5, color=col, mfc=col if r.p_holm < 0.05 else 'white', mew=1.2)
    ax.set_title(f'({"abc"[k]}) {title}', loc='left'); ax.set_xlabel('Difference in Dice')
    ax.set_xlim(-0.16, 0.22); ax.set_xticks([-0.1, 0, 0.1, 0.2]); ax.axhline(3.5, color='#999', lw=0.6, ls=':')
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([lab(c) for _, c in rows][::-1])
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', lw=1.4, label='F1, entanglement and function class'), Line2D([], [], color=ORANGE, marker='o', lw=1.4, label='F2, efficiency')], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.01))
fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1)); save(fig, 'fig_contrasts')

# ---------------------------------------------------------------- seeds (heatmap, vector)
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(wspace=0.06))
norm = mc.TwoSlopeNorm(vcenter=0, vmin=-0.05, vmax=0.05)
for k, (ax, fam, ttl) in enumerate(zip(axs, ['F1', 'F1_busbra'], ['BUSI', 'BUS-BRA'])):
    t = C[C.family == fam]
    M = np.array([[json.loads(r.per_seed)[str(s)] for s in range(5)] for _, r in t.iterrows()])
    pm = ax.pcolormesh(np.arange(6) - 0.5, np.arange(M.shape[0] + 1) - 0.5, M, cmap='RdBu_r', norm=norm, rasterized=False,
                       edgecolors='white', linewidth=1.0)
    ax.invert_yaxis()
    for i in range(M.shape[0]):
        for j in range(5):
            ax.text(j, i, f'{M[i, j]:.3f}'.replace('-', '−'), ha='center', va='center', fontsize=7.5,
                    color='white' if abs(M[i, j]) > 0.035 else 'black')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Seed {s}' for s in range(5)])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([lab(c) for c in t.contrast] if k == 0 else [])
    ax.axvline(2.5, color='black', lw=1.3, ls='--'); ax.set_title(f'({"ab"[k]}) {ttl}', loc='left', pad=20)
    ax.text(1, -0.62, 'Wave 1', ha='center', va='bottom', fontsize=8); ax.text(3.5, -0.62, 'Wave 2', ha='center', va='bottom', fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
cb = fig.colorbar(pm, ax=axs, fraction=0.03, pad=0.02); cb.set_label('Difference in Dice'); cb.outline.set_linewidth(0.5)
fig.savefig(os.path.join(OUT, 'fig_seeds.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig_seeds.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- fractions
fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)
mods = [('q4', BLUE, 'o'), ('tr4', '#2e8b57', 's'), ('cl4', '#666666', '^'), ('full', ORANGE, 'D')]
for k, (ax, (ds, title)) in enumerate(zip(axs, DS)):
    for m, col, mk in mods:
        ys, lo, hi = [], [], []
        for f in (0.1, 0.25, 0.5, 1.0):
            r = L[(L.dataset == ds) & (L.model == m) & (L.frac == f) & (L.plan == ('A' if f == 1 else 'C'))].iloc[0]
            ys.append(r.dice); lo.append(r.ci95_lo); hi.append(r.ci95_hi)
        x = np.array([10, 25, 50, 100])
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, ys, color=col, marker=mk, ms=4.5, lw=1.4, label=NAME[m])
    ax.set_xscale('log'); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(['10', '25', '50', '100']); ax.minorticks_off()
    ax.set_xlabel('Training images (%)'); ax.set_title(f'({"abc"[k]}) {title}', loc='left'); ax.set_ylim(0, 0.75)
axs[0].set_ylabel('Dice'); axs[0].legend(frameon=False, loc='upper left', handlelength=1.6)
fig.tight_layout(w_pad=0.6); save(fig, 'fig_fractions')
print('figures written to', OUT)
