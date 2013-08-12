
import numpy as np
from matplotlib import pylab as plt
from optparse import OptionParser
import os, sys

import sampling as smp
import features as ft
import data_processing as dp


def exp_simple_css(curve_path, resample_size, sigma_step, smoothing):
    """ exp_simple_css(curve_path, resample_size, sigma_step, smoothing)
    Simple CSS experiments
    """
    x, y = np.loadtxt(curve_path, unpack=True)
    c = np.array([x, y])
    rc = smp.resample_via_fft(c, resample_size)
    css = ft.generate_css(rc, 600, sigma_step)
    flt, mxs = ft.generate_visual_css(css, smoothing)
    return flt, mxs


def filtering(params):
    """docstring for filtering"""
    pass

def exp_gen_scss(dir_path, resample_size, sigma_step, smoothing):
    """ exp_gen_scss(dir_path, resample_size, sigma_step, smoothing)
    experimenting with scss
    """
    dpc = dp.ObjectModel()
    slices = dpc.initialize_from_slicefiles(dir_path)
    print 'Loaded ', len(slices), ' slice files'

    scss = np.zeros(shape=(len(slices), resample_size))  # hack
    maxs = np.zeros(shape=(len(slices), resample_size))  # hack

    for i, s in enumerate(slices):
        scss[i, :], mxs = exp_simple_css(s, resample_size+(smoothing-1), sigma_step, smoothing)
        ma = np.transpose( np.array(mxs) )
        mz = np.zeros(resample_size)
        mz[ma[0,:].tolist()] = ma[1,:] 
        maxs[i, :] = mz
        print 'Done with slice no: {0} out of {1}'.format(i,len(slices))

    return scss, maxs


def exp_eigen_css(curve_path, resample_size, sigma_step, return_all=False):
    """ exp_eigen_css(curve_path, resample_size, sigma_step, smoothing)
    Simple CSS experiments
    """
    x, y = np.loadtxt(curve_path, unpack=True)
    c = np.array([x, y])
    rc = smp.resample_via_fft(c, resample_size)
    css = ft.generate_css(rc, 600, sigma_step)
    ecss = ft.generate_eigen_css(css, return_all)
    return ecss



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="cpath",
                      help="PATH to a single curve slice", metavar="CPATH")
    parser.add_option("-d", "--dir", dest="dpath",
                      help="PATH to object curve slices", metavar="DPATH")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")
    parser.add_option("-t", "--type", default="css", dest="csstype",
                      help="types allowed: css, eigencss, [default: %default]")
    parser.add_option("-n", "--name", dest="scss_name",
                      help="Name for saving scss image", metavar="NAME")

    (options, args) = parser.parse_args(sys.argv)

    if len(args) != 1:
        print len(args), args
        print options
        parser.error('Incorrect number of arguiments, path and method needed')
    else:
        if options.cpath and options.csstype:
            if options.csstype == 'css':
                css = exp_simple_css(options.cpath, 400, 0.1, 5)   
                plt.plot(css)   
                plt.show(block=True)
            else:
                css = exp_eigen_css(options.cpath, 400, 0.1, False)   
                plt.plot(css)   
                plt.show(block=True)

        if options.dpath and options.csstype and options.scss_name:
            print 'scss will be saved as: {0}.npy'.format(options.scss_name)
            scss, maxs = exp_gen_scss(options.dpath, 300, .5, 10)
            np.save(options.scss_name, scss)
            np.save((options.scss_name + '_maxs'), maxs)


    # cpath = '../../../../../mThesis/code/branches/expdata/bunny_side/slice145.txt'
    # css1 = exp_simple_css(cpath, 400, 0.1, 5)
    # css2 = exp_simple_css(cpath, 400, 0.01, 3)
    # css3 = exp_simple_css(cpath, 400, 5, 2)

    # plt.plot(css1)
    # plt.plot(css2)
    # plt.plot(css3)



    # bunny_path = '../../../../../mThesis/code/branches/expdata/bunny/'
    # scss = exp_gen_scss(bunny_path, 300, 1, 3)
    # np.save('bunny_front', scss)

    # plt.show(block=True)
