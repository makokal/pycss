# curve evolution based on the CSS method

# from scipy import signal
import numpy as np
import gdevs as gd


def smooth_signal(signal, kernel):
    """ smooth_signal(signal, kernel)
    Smooth the given 1D signal by convolution with a specified kernel
    """
    return np.convolve(signal, kernel, mode='same')


def compute_curvature(curve, sigma):
    """ compute_curvature(curve, sigma)
    Compute the curvature of a 2D curve as given Mohkatarian
    and return the curvature signal at the given sigma

    Components of the 2D curve are:
    curve[0,:] and curve[1,:]
    """

    if curve[0,:].size < 2:
        raise Exception("Curve must have at least 2 points")

    sigx = curve[0,:]
    sigy = curve[1,:]
    g = gd.gaussian_kernel(sigma, 0, sigx.size, False)
    g_s = gd.gaussian_kernel(sigma, 1, sigx.size, False)
    g_ss = gd.gaussian_kernel(sigma, 2, sigx.size, False)

    X_s = smooth_signal(sigx, g_s)
    Y_s = smooth_signal(sigy, g_s)
    X_ss = smooth_signal(sigx, g_ss)
    Y_ss = smooth_signal(sigy, g_ss)

    kappa = ((X_s * Y_ss) - (X_ss * Y_s)) / (X_s**2 + Y_s**2)**(1.5)

    return kappa, smooth_signal(sigx, g), smooth_signal(sigy, g)
