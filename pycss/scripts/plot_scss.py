from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
plt.hold(True)

# load the data 
# scss = np.load('bunny_front.npy')
# scss = np.load('bunny_side.npy')
scss = np.load('cup1.npy')

a,b = scss.shape

x,y = np.meshgrid(np.arange(a),np.arange(b))
z = scss[x,y]

ax.plot_surface(x,y,z, cmap=cm.spectral)
# ax.plot_wireframe(x,y,z, cmap=cm.spectral)
# ax.scatter(x,y,z, cmap=cm.spectral)

ax.set_xlabel('slices')
ax.set_ylabel('t (curve parameter)')
ax.set_zlabel('Sigma (smoothing)')

plt.show()
