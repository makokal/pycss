
# simple 3d points plot


from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


data = np.loadtxt('../../../../../mThesis/datasets/experiments/cups/cup3.txt')


# set up figure and plot
fig = plt.figure()

ax = fig.gca(projection='3d')               # to work in 3d

plt.hold(True)

# plt.plot(data[:, 0], data[:, 1], data[:, 2], marker='.', ls='')
plt.plot(data[:, 0], data[:, 1], data[:, 2], ls='--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()