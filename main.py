# This is a sample Python script.
from batchnp import batch_np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from nn import simplenn
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

import numpy as np
from nnnp import simplenn


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # inputs = [1.0,2.0,3.0,2.5]
    # weights = [0.2,0.7,-0.5,1.0]
    # bias = 2.0
    #
    # outputs = np.dot(weights,inputs) + bias
    # print(outputs)
    # print(simplenn())

    a =[1,2,3]
    b =[2,3,4]

    a = np.array([a])
    b = np.array([b]).T
    print(np.dot(a,b))

    batch_np()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
