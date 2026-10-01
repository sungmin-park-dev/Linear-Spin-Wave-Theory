import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

COLORS = {'Y': '#2a78d6', 'UUD': '#eda100', 'V': '#1baf7a', 'Psi': '#e87ba4', 'P': '#d9d8d2',
          '2-site': '#eb6834', '4-site': '#4a3aa7', 'SkX': '#e34948', 'canted-1': '#008300'}
NAMES = {'Psi': 'Ψ', '2-site': '2-site stripe', '4-site': '4-site (Q=0)', 'SkX': '4-site SkX',
         'canted-1': 'uniform canted'}


def classify(p):
    c = p['candidates']
    pool = sorted(c, key=lambda x: x['classical'])
    e0 = pool[0]['classical']
    tied = [x for x in pool if x['classical'] < e0 + 1e-7]
    cw = min(tied, key=lambda x: (0 if x['label'] == 'P' else x['nsites']))
    if cw['status'] == 'stable':
        hw = p['harmonic_winner']
        return (hw['label'] if hw else cw['label']), False
    return cw['label'], True


def edges(values):
    v = np.array(sorted(set(values)))
    mid = (v[1:] + v[:-1]) / 2
    lo = np.r_[v[0] - (mid[0] - v[0]), mid]
    hi = np.r_[mid, v[-1] + (v[-1] - mid[-1])]
    return {float(a): (l, u) for a, l, u in zip(v, lo, hi)}


def panel(ax, pts, axis, ylabel):
    pts = [p for p in pts if p['axis'] == axis and p['h'] > 0]
    ex, ey = edges([p['h'] for p in pts]), edges([p['J'] for p in pts])
    for p in pts:
        lab, unstable = classify(p)
        (x0, x1), (y0, y1) = ex[p['h']], ey[p['J']]
        ax.add_patch(Rectangle((x0, y0*1e3), x1 - x0, (y1 - y0)*1e3, facecolor=COLORS[lab],
                               edgecolor='white', linewidth=0.6, alpha=0.35 if unstable else 1.0,
                               hatch='////' if unstable else None))
    ax.set_xlim(min(v[0] for v in ex.values()), max(v[1] for v in ex.values()))
    ax.set_ylim(min(v[0] for v in ey.values())*1e3, max(v[1] for v in ey.values())*1e3)
    ax.set_xlabel('h = g_z μ_B B (meV)')
    ax.set_ylabel(ylabel)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


d = json.load(open(sys.argv[1]))
for extra in sys.argv[3:]:
    d['points'] += json.load(open(extra))['points']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
panel(axes[0], d['points'], 'PD', 'J_PD (μeV), J_Γ = 0')
panel(axes[1], d['points'], 'Gamma', 'J_Γ (μeV), J_PD = 0')
axes[0].set_title('(a) h–J_PD', loc='left')
axes[1].set_title('(b) h–J_Γ', loc='left')
MU = 4.645 * 0.05788381806
for ax in axes:
    ax.axhline(10, color='#333', lw=0.8, ls=':')
    for B, name in ((0.2, 'Y 0.2 T'), (1.4, 'V 1.4 T')):
        ax.plot(B*MU, 10, 'o', ms=8, mfc='white', mec='#111', mew=1.5, zorder=5)
        ax.annotate(name, (B*MU, 10), xytext=(4, 6), textcoords='offset points', fontsize=8, color='#111')
seen = sorted({classify(p)[0] for p in d['points'] if p['h'] > 0}, key=list(COLORS).index)
handles = [Patch(facecolor=COLORS[k], label=NAMES.get(k, k)) for k in seen]
handles.append(Patch(facecolor='white', edgecolor='#555', hatch='////',
                     label='lowest classical state is LSWT-unstable'))
fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False, fontsize=9)
fig.suptitle('NBCP model, J = 0.075, J_z = 0.125 meV, S = 1/2: classical + LSWT zero-point, 1–4-site cells. '
             'Negative couplings map onto these by a global spin rotation: J_PD → −J_PD by R_z(π/2), J_Γ → −J_Γ by R_z(π).',
             fontsize=10)
fig.savefig(sys.argv[2], dpi=170)
