
import sampling as smp
import numpy as np
from matplotlib import pylab as plt

import features as ft
import data_processing as dp


def exp_simple_css(curve_path, resample_size, sigma_step, smoothing):
    """ exp_simple_css(curve_path, resample_size, sigma_step, smoothing)
    Simple CSS experiments
    """
    x, y = np.loadtxt(curve_path, unpack=True)
    c = np.array([x, y])
    rc = smp.resample_via_fft(c, resample_size)
    css, lss = ft.generate_css(rc, 600, sigma_step)
    flt = ft.generate_visual_css(css, smoothing)
    return flt


def exp_gen_scss(dir_path, resample_size, sigma_step, smoothing):
    """ exp_gen_scss(dir_path, resample_size, sigma_step, smoothing)
    experimenting with scss
    """
    dpc = dp.ObjectModel()
    slices = dpc.initialize_from_slicefiles(dir_path)
    print 'Loaded ', len(slices), ' slice files'

    scss = np.zeros(shape=(len(slices), resample_size - 2))  # hack

    for i, s in enumerate(slices):
        scss[i, :] = exp_simple_css(s, resample_size, sigma_step, smoothing)
        print 'Done with slice no: {0} out of {1}'.format(i,len(slices))

    return scss


if __name__ == '__main__':


    # cpath = '../../../../../mThesis/code/branches/expdata/bunny_side/slice145.txt'
    # css1 = exp_simple_css(cpath, 400, 0.1, 5)
    # css2 = exp_simple_css(cpath, 400, 0.01, 3)
    # css3 = exp_simple_css(cpath, 400, 5, 2)

    # plt.plot(css1)
    # plt.plot(css2)
    # plt.plot(css3)

    bunny_path = '../../../../../mThesis/code/branches/expdata/bunny/'
    scss = exp_gen_scss(bunny_path, 300, 1, 3)
    np.save('bunny_front', scss)

    plt.show(block=True)