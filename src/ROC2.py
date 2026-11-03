import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

models = ['Véletlen erdő', 'XGBoost', 'GraphSAGE', 'Hibrid']
auc_pr  = [0.9430, 0.9434, 0.8708, 0.9466]
auc_roc = [0.9987, 0.9988, 0.9981, 0.9989]
cost    = [25737, 25163, 30974, 25235]

colors = ['#EBD4D8', '#DDB7BE', '#C98C98', '#8C3D52']

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 1])

for ax, vals, title, fmt in zip(
    [ax1, ax2, ax3],
    [auc_pr, auc_roc, cost],
    ['AUC-PR', 'AUC-ROC', 'Teszt adathalmazon mért költség (€)'],
    ['{:.4f}', '{:.4f}', '{:,.0f}']
):
    bars = ax.bar(models, vals, color=colors)

    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)

    ymin = min(vals) * 0.98
    ymax = max(vals) * 1.02
    ax.set_ylim(ymin, ymax)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (ymax - ymin) * 0.01,
            fmt.format(v),
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.savefig('modellek_osszehasonlitas2.png', dpi=300, bbox_inches='tight')
plt.show()