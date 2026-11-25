"""Readable tables for the manuscript (main text) and the Supplementary Material.

Main text: model levels (Dice only), F1/F2 contrasts in long format, F3/F4 contrasts on BUSI, audit (means only).
Supplementary: the same with all 95% (and 90%) intervals, all external families, and the exploratory gap analysis.
Usage: RESULTS=<folder with the analysis CSVs> DECISION=<plan_decision.json> OUT=<folder> python tabs2.py
"""
import os, json
import numpy as np, pandas as pd

R = os.environ.get('RESULTS', 'results/')
OUT = os.environ.get('OUT', '.')
C = pd.read_csv(R + 'contrasts.csv'); L = pd.read_csv(R + 'model_levels.csv'); A = pd.read_csv(R + 'audit.csv')
K = pd.read_csv(R + 'costs.csv'); N = pd.read_csv(R + 'normals_fp.csv')
G = pd.read_csv(R + 'gap_contrasts.csv') if os.path.exists(R + 'gap_contrasts.csv') else None
dec = json.load(open(os.environ.get('DECISION', 'plan_decision.json')))['selected']
NAME = {'q': 'Q', 'qp': 'Qp', 's': 'S', 'tr': 'Tr', 'cl': 'Cl', 'id': 'Id'}
DSN = {'busi': 'BUSI', 'busbra': 'BUS-BRA', 'breast': 'BrEaST'}


def nm(m):
    if m == 'full': return 'Full'
    if m == 'unet': return 'U-Net'
    b = m.rstrip('248'); return f'{NAME[b]}$_{m[len(b):]}$'


def f3(x): return f'{x:.3f}'.replace('-', '$-$')
def ci(lo, hi): return f'{f3(lo)} to {f3(hi)}'
def pf(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
def pair(c): a, b = [x.strip() for x in c.split('-')]; return f'{nm(a)} vs {nm(b)}'
VER = {'equivalent': 'Equivalent', 'inconclusive': 'Inconclusive', 'difference (higher)': 'Higher',
       'difference (lower)': 'Lower', 'non-inferior': 'Noninferior'}
def ver(v): return ', '.join(VER[x.strip()] for x in v.split(','))
def write(name, s): open(os.path.join(OUT, name), 'w').write(s)


# ------------------------------------------------------------------ main: model levels (Dice only)
order = ['q4', 'qp4', 's4', 'tr4', 'cl4', 'id4', 'full', 'unet']
lines = []
for m in order:
    g = lambda ds: L[(L.dataset == ds) & (L.model == m) & (L.plan == 'A')].iloc[0]
    k = K[(K.plan == 'A') & (K.model == m)].iloc[0]; fp = N[(N.plan == 'A') & (N.model == m)].iloc[0]
    cells = [nm(m), str(int(k.block)), f'{k.trainable / 1e6:.2f}', str(dec.get(m, '--'))]
    cells += [f'{g(ds).dice:.3f}' for ds in ('busi', 'busbra', 'breast')]
    cells += [f"{100 * g('busi').empty_rate:.1f}", f'{100 * fp.fp_rate:.0f}', f'{k.minutes:.0f}', f'{k.infer_ms_per_image:.0f}']
    lines.append(' & '.join(cells) + r' \\')
    if m == 'id4': lines.append(r'\midrule')
write('tab_levels.tex', r'''\begin{table}[!t]
\centering
\caption{Mean Dice of the eight models with four values in the slot and all training images (plan A, five seeds). The 95\% intervals are given in Supplementary Table S1.}\label{tab:levels}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{Weights} & \multicolumn{3}{c}{Dice} & \multicolumn{2}{c}{BUSI errors (\%)} & \multicolumn{2}{c}{Cost} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Model & Block & All (M) & $k$ & BUSI & BUS-BRA & BrEaST & Empty & FP & Min & ms \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}

\smallskip\parbox{\linewidth}{\scriptsize $k$, factor of the block learning rate. Empty, lesion images without any predicted pixel. FP, normal images with a predicted lesion. Min, training time per run. ms, inference time per image on a T4 GPU.}
\end{table}
''')

# ------------------------------------------------------------------ main: F1 and F2 in long format
rows = [('F1', 'q4 - qp4'), ('F1', 'q4 - s4'), ('F1', 'q4 - tr4'), ('F1', 'q4 - cl4'),
        ('F2', 'tr4 - full'), ('F2', 'q4 - full'), ('F2', 'tr4 - unet'), ('F2', 'q4 - unet')]
lines = []
for ds in ('busi', 'busbra', 'breast'):
    lines.append(r'\multicolumn{6}{@{}l}{\textit{' + DSN[ds] + (' (internal)' if ds == 'busi' else ' (external)') + r'}} \\')
    for f, c in rows:
        r = C[(C.family == (f if ds == 'busi' else f'{f}_{ds}')) & (C.contrast == c)].iloc[0]
        lines.append(f"{f} & {pair(c)} & {f3(r['diff'])} & {ci(r.ci95_lo, r.ci95_hi)} & {pf(r.p_holm)} & {ver(r.verdict)} \\\\")
    if ds != 'breast': lines.append(r'\addlinespace')
write('tab_contrasts.tex', r'''\begin{table}[!t]
\centering
\caption{Primary (F1) and efficiency (F2) contrasts. Difference, mean difference in Dice (first model minus second). $p$, Holm adjusted within each family and dataset.}\label{tab:contrasts}
\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrcrl@{}}
\toprule
Family & Contrast & Difference & 95\% interval & $p$ & Verdict \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}

\smallskip\parbox{\linewidth}{\scriptsize Equivalent, 90\% interval inside $\pm0.02$. Noninferior, one sided test against $-0.02$. Higher or Lower, Holm adjusted $p<0.05$. Inconclusive, none of these.}
\end{table}
''')

# ------------------------------------------------------------------ main: F3 and F4 on BUSI
lines = []
for fam, label in (('F3_p010', '10\\%'), ('F3_p025', '25\\%'), ('F3_p050', '50\\%')):
    lines.append(r'\multicolumn{5}{@{}l}{\textit{F3, ' + label + r' of the training images}} \\')
    for _, r in C[C.family == fam].iterrows():
        lines.append(f"{pair(r.contrast)} & {f3(r['diff'])} & {ci(r.ci95_lo, r.ci95_hi)} & {pf(r.p_holm)} & {ver(r.verdict)} \\\\")
    lines.append(r'\addlinespace')
lines.append(r'\multicolumn{5}{@{}l}{\textit{F4, two and eight qubits}} \\')
for _, r in C[C.family == 'F4'].iterrows():
    lines.append(f"{pair(r.contrast)} & {f3(r['diff'])} & {ci(r.ci95_lo, r.ci95_hi)} & {pf(r.p_holm)} & {ver(r.verdict)} \\\\")
write('tab_secondary.tex', r'''\begin{table}[!t]
\centering
\caption{Contrasts on BUSI with fewer training images (F3, plan C) and with two or eight qubits (F4, plan B). Columns as in Table~\ref{tab:contrasts}; the external sets are in Supplementary Table S2.}\label{tab:secondary}
\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrcrl@{}}
\toprule
Contrast & Difference & 95\% interval & $p$ & Verdict \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ------------------------------------------------------------------ main: audit (means)
lines = []
for m in ['q2', 'qp2', 's2', 'tr2', 'cl2', 'q4', 'qp4', 's4', 'tr4', 'cl4', 'id4', 'q8', 'qp8', 's8', 'tr8', 'cl8']:
    r = A[A.model == m].iloc[0]
    g = lambda c, d=3: ('--' if c not in r or pd.isna(r[c]) else f'{r[c]:.{d}f}')
    mw = '0' if m.startswith('qp') else g('entanglement_mw', 2)
    lines.append(' & '.join([nm(m), str(dec.get(m, '--')), g('drop_const'), g('drop_swap'), g('change_block'),
                             g('grad_share_final'), mw if m[0] == 'q' else '--', g('surrogate_r2') if m[0] == 'q' else '--']) + r' \\')
    if m in ('cl2', 'id4'): lines.append(r'\midrule')
write('tab_audit.tex', r'''\begin{table}[!t]
\centering
\caption{Bottleneck audit on BUSI, mean over 25 runs per model. The 95\% intervals are given in Supplementary Table S3.}\label{tab:audit}
\footnotesize\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{Dice drop} & \multicolumn{2}{c}{Block weights} & \multicolumn{2}{c}{Circuit} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
Model & $k$ & Mean code & Other image & Change & Gradient & MW & $R^2$ \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}

\smallskip\parbox{\linewidth}{\scriptsize Dice drop, loss of Dice when the code is replaced by the mean training code or by the code of another training image. Change, relative change of the block weights from their start. Gradient, share of the gradient norm on the block weights at the restored epoch. MW, Meyer--Wallach entanglement. $R^2$, fit of the Fourier surrogate to the trained circuit on test codes.}
\end{table}
''')

# ------------------------------------------------------------------ supplementary tables
def longtab(label, caption, head, body, spec):
    return (r'\begin{longtable}{' + spec + '}\n\\caption{' + caption + r'}\label{' + label + r'}\\' + '\n\\toprule\n' + head +
            r' \\' + '\n\\midrule\n\\endfirsthead\n\\toprule\n' + head + r' \\' + '\n\\midrule\n\\endhead\n' + '\n'.join(body) +
            '\n\\bottomrule\n\\end{longtable}\n')

body = []
for (ds, plan, frac), g in L.groupby(['dataset', 'plan', 'frac'], sort=False):
    pass
for ds in ('busi', 'busbra', 'breast'):
    for plan, frac in (('A', 1.0), ('B', 1.0), ('C', 0.5), ('C', 0.25), ('C', 0.1)):
        t = L[(L.dataset == ds) & (L.plan == plan) & (L.frac == frac)]
        if t.empty: continue
        for _, r in t.iterrows():
            body.append(f"{DSN[ds]} & {plan} & {int(frac * 100)} & {nm(r.model)} & {r.dice:.3f} & {ci(r.ci95_lo, r.ci95_hi)} & {r.seed_sd:.3f} & {100 * r.empty_rate:.1f} \\\\")
s1 = longtab('stab:levels', 'Mean Dice of every model with 95\\% interval, standard deviation over seeds and share of empty predictions.',
             r'Dataset & Plan & Data (\%) & Model & Dice & 95\% interval & SD seeds & Empty (\%)', body, '@{}lllcrcrr@{}')

def famlab(f):
    parts = f.split('_'); ds = 'busi'
    if parts[-1] in ('busbra', 'breast'): ds = parts.pop()
    fr = {'p010': ' (10%)', 'p025': ' (25%)', 'p050': ' (50%)'}.get(parts[1], '') if len(parts) > 1 else ''
    return f"{parts[0]}{fr.replace('%', chr(92) + '%')}, {DSN[ds]}"


body = []
for fam in C.family.unique():
    for _, r in C[C.family == fam].iterrows():
        body.append(f"{famlab(fam)} & {pair(r.contrast)} & {f3(r['diff'])} & {ci(r.ci95_lo, r.ci95_hi)} & {ci(r.ci90_lo, r.ci90_hi)} & {pf(r.p)} & {pf(r.p_holm)} & {ver(r.verdict)} \\\\")
s2 = longtab('stab:contrasts', 'All prespecified contrasts on the three datasets. Raw and Holm adjusted $p$ values.',
             r'Family & Contrast & Difference & 95\% interval & 90\% interval & $p$ & Holm $p$ & Verdict', body, '@{}llrccrrl@{}')

body = []
for _, r in A.iterrows():
    def c3(c): return '--' if c not in r or pd.isna(r[c]) else f"{r[c]:.3f} ({f3(r[c + '_lo'])} to {f3(r[c + '_hi'])})" if c + '_lo' in r and not pd.isna(r[c + '_lo']) else f'{r[c]:.3f}'
    body.append(f"{nm(r.model)} & {c3('drop_const')} & {c3('drop_swap')} & {c3('change_block')} & {c3('grad_share_final')} \\\\")
s3 = longtab('stab:audit', 'Bottleneck audit with 95\\% intervals over runs.',
             r'Model & Drop, mean code & Drop, other image & Weight change & Gradient share', body, '@{}lcccc@{}')

s4 = ''
if G is not None:
    body = [f"{DSN[r.external]} & {pair(r.contrast)} & {f3(r.gap_diff)} & {ci(r.ci95_lo, r.ci95_hi)} & {pf(r.p)} & {pf(r.p_holm)} \\\\" for _, r in G.iterrows()]
    s4 = longtab('stab:gap', 'Exploratory analysis, not in the protocol: difference between two models in the loss of Dice from BUSI to an external set (contrast on BUSI minus contrast on the external set).',
                 r'External set & Contrast & Difference & 95\% interval & $p$ & Holm $p$', body, '@{}llrcrr@{}')
write('supp_tables.tex', s1 + '\n' + s2 + '\n' + s3 + '\n' + s4)
print('tables written to', OUT)
