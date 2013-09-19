import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.ndimage import filters

bunny_front_raw = np.load('bunny_front.npy')
bunny_side_raw = np.load('bunny_side.npy')

# minor smothing 
ksize = 5
bf = filters.gaussian_filter(bunny_front_raw, ksize)
bs = filters.gaussian_filter(bunny_side_raw, ksize)

# plots 
plt.figure('Front Slicing')
plt.imshow(bf, cmap='spectral', origin='lower')
plt.title('Front Slicing')

plt.figure('Side Slicing')
plt.imshow(bs, cmap='spectral', origin='lower')
plt.title('Side Slicing')

plt.show(block=True)