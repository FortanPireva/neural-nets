import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

x, y = vertical_data(samples=100,classes=3)

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='brg')
plt.show()

