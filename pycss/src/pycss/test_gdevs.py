

import gdevs as gd
import evolution as ev
import features as ft

from matplotlib import pylab as plt
import numpy as np


def test_evolution():
	# make a basic synthetic curve
	curve = np.zeros(shape=(2,400))
	t = np.linspace(-4,4,400)

	curve[0,:] = 5*np.cos(t) - np.cos(6*t)
	curve[1,:] = 15*np.sin(t) - np.sin(6*t)

	# test css
	css,lss = ft.generate_css(curve, 600, 0.1)
	flt = ft.generate_visual_css(css, 5)
	# plt.plot(lss[0,:], lss[1,:], marker='s',color='r', ls='')
	# plt.pcolor(css)
	plt.plot(flt)

	# test evolve the curve
	# kappa,xx,yy = ev.compute_curvature(curve, 3)
	# xs = ft.find_zero_crossings(kappa)
	# print xs
	# plt.plot(kappa)
	# plt.plot(kappa[xs])
	# plt.plot(curve[0,:], curve[1,:])

	# plt.plot(curve[0,xs], curve[1,xs], marker='o',color='r', ls='')
	# plt.plot(xx, yy)
	plt.show()



if __name__ == '__main__':
	g0,t0 = gd.gaussian_kernel(7,0,10,True)
	g1,t1 = gd.gaussian_kernel(7,1,None,True)
	g2,t2 = gd.gaussian_kernel(7,2,None,True)

	# print g0
	# print np.random.rand(1,10)

	# plt.plot(t0,g0)
	# plt.plot(t1,g1)
	# plt.plot(t2,g2)
	# plt.show()
	test_evolution()