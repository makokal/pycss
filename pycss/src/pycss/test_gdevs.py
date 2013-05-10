

import gdevs as gd
from matplotlib import pylab as plt


if __name__ == '__main__':
	g0,t0 = gd.gaussian_kernel(7,0,None,True)
	g1,t1 = gd.gaussian_kernel(7,1,None,True)
	g2,t2 = gd.gaussian_kernel(7,2,None,True)
	plt.plot(t0,g0)
	plt.plot(t1,g1)
	plt.plot(t2,g2)
	plt.show()