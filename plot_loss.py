import numpy as np
from matplotlib import pyplot as plt

loss = np.load('result/loss_all.npy')

plt.figure()
plt.plot(np.arange(loss.shape[0]), loss)
plt.show()
