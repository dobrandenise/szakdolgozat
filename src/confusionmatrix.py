import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

cms = {
    'Véletlen erdő': [[118067, 687], [76, 1352]],
    'XGBoost':       [[118088, 666], [77, 1351]],
    'GraphSAGE':     [[117880, 874], [83, 1345]],
    'Hibrid':        [[118062, 692], [74, 1354]],
}

cmap = LinearSegmentedColormap.from_list(
    "burgundy",
    ["#F7F0F2",
     "#E7C8D0",
     "#D3A0AD",
     "#B96A80",
     "#8E3D57"]
)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
labels = ['Jogszerű', 'Csalárd']

for ax, (name, cm) in zip(axes.flat, cms.items()):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Modell előrejelzése'); ax.set_ylabel('Valódi')
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}',
                    ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=9)

plt.tight_layout()
plt.savefig('konfuzios_matrixok.png', dpi=300, bbox_inches='tight')
plt.show()