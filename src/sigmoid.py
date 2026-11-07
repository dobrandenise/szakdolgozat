import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-6, 6, 300)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(x, y, color='#8C3D52', linewidth=2)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('z (nyers modellkimenet)')
ax.set_ylabel('σ(z)')
ax.set_title('Szigmoid aktivációs függvény')
ax.set_yticks([0, 0.5, 1])
ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
plt.tight_layout()
plt.savefig('sigmoid.png', dpi=300, bbox_inches='tight')
plt.show()