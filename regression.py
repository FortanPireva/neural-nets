import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

x, y = sine_data()

plt.plot(x,y)
plt.show()