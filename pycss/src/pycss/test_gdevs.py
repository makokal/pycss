

import gdevs as gd
# import evolution as ev
import features as ft
import data_processing as dp
import sampling as smp

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy.signal as sig


TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)


def turn(p, q, r):
    return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)


def _keep_left(hull, r):
    while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
    if not len(hull) or hull[-1] != r:
        hull.append(r)
    return hull

 
def convex_hull(points):
    """Returns points on convex hull of an array of points in CCW order."""
    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in xrange(1, len(u) - 1)) or l


def test_evolution():

    # x,y = np.loadtxt('../../../../../mThesis/datasets/experiments/boxes/ex5/slice150.txt', unpack=True)
    # c = np.array([x, y])
    
    # # cart to polar then sort then back
    # r = np.sqrt(np.square(c[0, :]) + np.square(c[1, :]))
    # theta = np.arctan2(c[1, :], c[0, :])

    # pa = np.vstack((r, theta))
    # print pa.shape, r.shape, theta.shape
    # pas = pa[pa[:, 1].argsort()]

    # ctx = pas[0, :] * np.cos(pas[1, :])
    # cty = pas[0, :] * np.sin(pas[1, :])
    # ct = np.vstack((ctx, cty))


    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='#d5de9c')
    # ax.plot(pas[1, :], pas[0, :], color='#ee8d18', lw=3)

    # rc = smp.resample_via_fft(c, 302)

    # plt.plot(c[0,:], c[1,:], marker='o',color='r', ls='')
    # plt.plot(c[0,:], c[1,:], color='#ee8d18', lw=3)
    # plt.plot(ct[0,:], ct[1,:])

    # plt.plot(rc[0,:], rc[1,:])
    # # plt.plot(h[0,:], f[1,:])
    # # plt.plot(c[0,:])
    # # plt.plot(h[0,:])
    # # plt.plot(h[1,:])

    # make a basic synthetic curve
    curve = np.zeros(shape=(2, 50))
    t = np.linspace(-4, 4, 50)

    curve[0,:] = 5*np.cos(t) - np.cos(6*t)
    curve[1,:] = 15*np.sin(t) - np.sin(6*t)

    rc = smp.resample_via_fft(curve, 400)

    # # test scss continuity
    # scss = np.zeros(shape=(300, 300))  # hack

    # for i in range(300):
    #     css, lss = ft.generate_css(rc, 600, 1)
    #     scss[i, :] = ft.generate_visual_css(css, 3)
    #     print 'Done with slice no: {0} out of {1}'.format(i, 300)

    # np.save('synthetic', scss)

    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')
    plt.plot(rc[0,:], rc[1,:])

    # css = ft.generate_css(rc, 600, .01)
    # vis, m = ft.generate_visual_css(css, 3)
    # plt.plot(vis)
    # mv = np.array(m)
    # plt.plot( mv[:,0], mv[:,1], marker='o',color='r', ls='')
    # plt.legend(['CSS curve','Maxima'])
    # plt.xlabel('t (Curve parameter) ')
    # plt.ylabel(r'$\sigma$, evolution')

    plt.show(block=True)


def test_3D():
    # from numpy import pi, sin, cos, mgrid
    # dphi, dtheta = pi/250.0, pi/250.0
    # [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    # m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    # r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    # x = r*sin(phi)*cos(theta)
    # y = r*cos(phi)
    # z = r*sin(phi)*sin(theta)

    # x,y,z = np.loadtxt('../../../../../mThesis/datasets/banana.txt', unpack=True)
    # points = np.array([x,y,z])


    # dpc = dp.ObjectModel(points)
    # c = dpc.cut_clice_curve(9.3, 2)

    x, y = np.loadtxt('../../../../../mThesis/code/branches/expdata/banana/slice155.txt', unpack=True)
    c = np.array([x, y])
    # rc = smp.resample_curve(c, 400, 0.1, True)
    rc = smp.resample_via_fft(c, 400)

    plt.plot(c[0,:],c[1,:])
    plt.plot(rc[0,:],rc[1,:])

    # css,lss = ft.generate_css(c, 600, 0.1)
    # flt = ft.generate_visual_css(css, 2)

    # plt.plot(flt)
    # plt.plot(c[0,:], c[1,:], marker='o',color='r', ls='')
    plt.show()


    # plot the data
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x,y,z)
    # plt.show()

def test_slicefiles():
    # load the data from slice files
    dpc = dp.ObjectModel()
    slices = dpc.initialize_from_slicefiles('../../../../../mThesis/code/branches/expdata/banana/')
    print 'Loaded ', len(slices), ' files'

    # go over the slices to generate SCSS
    x, y = np.loadtxt(slices[150], unpack=True)
    s = UnivariateSpline(x, y, k=5, s=0)
    xs = np.linspace(x.min(), x.max(), 400)
    ys = s(xs)

    print ys
    # plt.plot(x,y)
    plt.plot(xs, ys)
    plt.show()

    # c = np.array([x,y])
    # css,lss = ft.generate_css(c, 600, 0.01)
    # flt = ft.generate_visual_css(css, 2)
    # plt.plot(flt)
    # plt.show()



if __name__ == '__main__':
    # g0, t0 = gd.gaussian_kernel(7, 0, 10, True)
    # g1, t1 = gd.gaussian_kernel(7, 1, None, True)
    # g2, t2 = gd.gaussian_kernel(7, 2, None, True)

    # print g0
    # print np.random.rand(1,10)

    # plt.plot(t0,g0)
    # plt.plot(t1,g1)
    # plt.plot(t2,g2)
    # plt.show()

    test_evolution()
    # test_3D()
    # test_slicefiles()
