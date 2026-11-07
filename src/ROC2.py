import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

models = ['Véletlen erdő', 'XGBoost', 'GraphSAGE', 'Hibrid']
auc_pr  = [0.94302, 0.94314, 0.86106, 0.94672]
auc_roc = [0.99872, 0.99877, 0.99806, 0.99892]
cost    = [25736.53, 25649.64, 30096.90, 24952.57]

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
    ['{:.4f}', '{:.4f}', '{:.2f}']
):
    bars = ax.bar(models, vals, color=colors)

    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)

    if title == 'AUC-ROC':
        ymin = min(vals) * 0.98
        ymax = 1.001
        ax.set_yticks([0.98, 0.985, 0.99, 0.995, 1.0])
    else:
        ymin = min(vals) * 0.98
        ymax = max(vals) * 1.02
    ax.set_ylim(ymin, ymax)

    if title in ('AUC-PR', 'AUC-ROC'):
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f'{value:.4f}'.replace('.', ','))
        )
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f'{value:.2f}'.replace('.', ','))
        )

    for bar, v in zip(bars, vals):
        label = (
            fmt.format(v).replace('.', ',')
            if title in ('AUC-PR', 'AUC-ROC')
            else fmt.format(v).replace('.', ',')
        )
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (ymax - ymin) * 0.01,
            label,
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.savefig('modellek_osszehasonlitas2.png', dpi=300, bbox_inches='tight')
plt.show()