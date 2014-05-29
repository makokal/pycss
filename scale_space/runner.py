

from css import CurvatureScaleSpace, SlicedCurvatureScaleSpace
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sg
from sklearn.mixture import GMM, DPGMM


def simple_signal(np_points):
    curve = np.zeros(shape=(2, np_points))
    t = np.linspace(-4, 4, np_points)

    curve[0, :] = 5 * np.cos(t) - np.cos(6 * t)
    curve[1, :] = 15 * np.sin(t) - np.sin(6 * t)

    return curve

def some_function(some_args):
    """ 
    :Parameters:
    """
    pass


def run():
    curve = simple_signal(np_points=400)
    c = CurvatureScaleSpace()
    cs = c.generate_css(curve, curve.shape[1], 0.01)
    vcs = c.generate_visual_css(cs, 9)
    # ecs = c.generate_eigen_css(cs)
    # print ecs.shape

    plt.figure('Sample Curve')
    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')

    plt.figure('CSS')
    plt.plot(vcs)

    # plt.figure('EigenCSS')
    # plt.plot(ecs)

    plt.show()

if __name__ == '__main__':
    run()
