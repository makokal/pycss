# CSS relared features computation

import numpy as np
import evolution as ev
import gdevs as gd


def find_zero_crossings(kappa):
    """ find_zero_crossings(kappa)
    Locate the zero crossing points of the curvature kappa
    """

    crossings = []

    for i in range(0, kappa.size - 2):
        if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or \
                (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
            crossings.append(i)

    return crossings


def generate_css(curve, max_sigma, step_sigma):
    """ generate_css(curve, max_sigma, step_sigma)
    Generates a CSS image representation by repetatively smoothing the initial curve L_0 with increasing sigma
    """

    cols = curve[0, :].size  # width
    rows = max_sigma / step_sigma  # height

    css = np.zeros(shape=(rows, cols))
    # csslist = np.zeros(shape=(2, rows))

    srange = np.linspace(1, max_sigma - 1, rows)
    for i, sigma in enumerate(srange):
        # compute curvature
        kappa, sx, sy = ev.compute_curvature(curve, sigma)

        # find interest points
        xs = find_zero_crossings(kappa)
        # print 'evolution step ', i, sigma, len(xs)

        # save the interest points
        if len(xs) > 0 and sigma < max_sigma - 1:
            for c in xs:
                css[i, c] = sigma  # change to any positive 
                # csslist[0, i], csslist[1, i] = c, sigma

        else:
            return css


def generate_visual_css(rawcss, closeness):
    """ generate_visual_css(rawcss, closeness)
    Generate a 1D signal that can be plotted to depict the CSS by taking
    column maximums. Further checks for close interest points and nicely
    smooths them with weighted moving average
    """

    flat_signal = np.amax(rawcss, axis=0)

    # minor smoothing via moving averages
    window = closeness
    # weights = np.repeat(1.0, window) / window # uniform weights
    weights = gd.gaussian_kernel(window, 0, window, False)  # gaussian weights
    sig = np.convolve(flat_signal, weights)[window - 1:-(window - 1)]

    maxs = []

    # get maximas
    w = sig.size
    print w

    for i in range(1, w - 1):
        if sig[i - 1] < sig[i] and sig[i] > sig[i + 1]:
            maxs.append([i, sig[i]])

    return sig, maxs


def generate_eigen_css(rawcss, return_all=False):
    """ generate_eigen_css(rawcss, return_all)
    Generates Eigen-CSS features
    """
    rowsum = np.sum(rawcss, axis=0)
    csum = np.sum(rawcss, axis=1)

    # hack to trim c
    # limc = np.amax(csum, axis=None)
    colsum = csum[0:rowsum.size]
    # colsum = csum

    freq = np.fft.fft(rowsum)
    mag = abs(freq)

    tilde_rowsum = np.fft.ifft(mag)

    feature = np.concatenate([tilde_rowsum, colsum], axis=0)

    if not return_all:
        return feature
    else:
        return feature, rowsum, tilde_rowsum, colsum
