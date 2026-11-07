"""Make the summary figures used in the report from the result CSVs."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#1f2328', '#59636e', '#d8dee4'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})


def smd_per_machine(out):
    df = pd.read_csv('results/smd/smd_results.csv').sort_values('F1_PA', ascending=False)
    x = np.arange(len(df))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(x - w / 2 - 0.01, df.F1_PA, w, color=BLUE, label='with point adjustment')
    ax.bar(x + w / 2 + 0.01, df.F1_raw, w, color=ORANGE, label='without point adjustment')
    ax.axhline(0.9382, color=INK, lw=1, ls='--')
    ax.text(len(df) - 0.5, 0.955, 'paper, mean F1 = 0.938 (with PA)', ha='right', va='bottom',
            fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('machine-', '') for m in df.machine], rotation=90, fontsize=7)
    ax.set_ylabel('best F1')
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{out}/fig_smd_per_machine.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/fig_smd_per_machine.png', dpi=200, bbox_inches='tight')


def uci_and_judge(out):
    bt = pd.read_csv('results/uci/uci_detection_by_type.csv', index_col=0)
    order = ['dropout', 'drift', 'decoupling', 'spike']
    bt = bt.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, bt['high'], w, color=BLUE, label='high severity')
    ax.bar(x + w / 2 + 0.01, bt['low'], w, color=ORANGE, label='low severity')
    for i, t in enumerate(order):
        for dx, col in [(-w / 2, 'high'), (w / 2, 'low')]:
            ax.text(i + dx, bt.loc[t, col] + 0.03, f'{bt.loc[t, col]:.2f}', ha='center',
                    fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('share of segments detected')
    ax.set_title('(a) USAD detection by injected type', fontsize=9, loc='left', color=INK)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=7, loc='upper right')

    s = json.load(open('results/llm_judge/judge_summary.json'))
    ax = axes[1]
    vals = [s['usad_alert_precision'], s['usad_plus_judge_precision']]
    labels = ['USAD alerts', 'USAD + LLM judge']
    ax.bar([0, 1], vals, 0.55, color=[MUTED, AQUA])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('alert precision')
    ax.set_title(f'(b) Precision of 110 alerts (kappa = {s["cohen_kappa"]:.2f})', fontsize=9,
                 loc='left', color=INK)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(f'{out}/fig_uci_type_and_judge.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/fig_uci_type_and_judge.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    smd_per_machine('figures')
    uci_and_judge('figures')
