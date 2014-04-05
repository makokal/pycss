

import numpy as np


def _gaussian_kernel(sigma, order, t):
    """ _gaussian_kernel(sigma, order, t)
    Calculate a Gaussian kernel of the given sigma and with the given
    order, using the given t-values.
    """

    # if sigma 0, kernel is a single 1.
    if sigma == 0:
        return np.array([1.0])

    # pre-calculate some stuff
    sigma2 = sigma ** 2
    sqrt2 = np.sqrt(2)

    # Calculate the gaussian, it is unnormalized. We'll normalize at the end.
    basegauss = np.exp(- t ** 2 / (2 * sigma2))

    # Scale the t-vector, what we actually do is H( t/(sigma*sqrt2) ),
    # where H() is the Hermite polynomial.
    x = t / (sigma * sqrt2)

    # Depending on the order, calculate the Hermite polynomial already generated
    # from mathematica
    if order < 0:
        raise Exception("The order should not be negative!")
    elif order == 0:
        part = 1
    elif order == 1:
        part = 2 * x
    elif order == 2:
        part = -2 + 4 * x ** 2
    else:
        raise Exception("Order above 2 is not implemented!")

    # Apply Hermite polynomial to gauss
    k = (-1) ** order * part * basegauss

    # By calculating the normalization factor by integrating the gauss, rather
    # than using the expression 1/(sigma*sqrt(2pi)), we know that the KERNEL
    # volume is 1 when the order is 0.
    norm_default = 1 / basegauss.sum()
    #           == 1 / ( sigma * sqrt(2*pi) )

    # Here's another normalization term that we need because we use the
    # Hermite polynomials.
    norm_hermite = 1 / (sigma * sqrt2) ** order

    # Normalize and return
    return k * (norm_default * norm_hermite)


def gaussian_kernel(sigma, order=0, N=None, returnt=False):
    """ gaussian_kernel(sigma, order, N, returnt)
    Compute the gaussian kernel given a width and derivative order and optionally
    the length.

    Parameters
    -------------
    sigma : float
        Width of the Gaussian kernel
    order : int
        Derivative order of the kernel
    N : int, optional
        Number of samples to return
    returnt : Bool
        Whether or not to return the abscissa

    Returns
    -----------
    k : float
        The samples
    t : float
        Sample indices

    """

    # checking inputs
    if not N:
        # Calculate ratio that is small, but large enough to prevent errors
        ratio = 3 + 0.25 * order - 2.5 / ((order - 6) ** 2 + (order - 9) ** 2)
        # Calculate N
        N = int(np.ceil(ratio * sigma)) * 2 + 1

    elif N > 0:
        if not isinstance(N, int):
            N = int(np.ceil(N))

    elif N < 0:
        N = -N
        if not isinstance(N, int):
            N = int(np.ceil(N))
        N = N * 2 + 1

    # Check whether given sigma is large enough
    sigmaMin = 0.5 + order ** (0.62) / 5
    if sigma < sigmaMin:
        print('WARNING: The scale (sigma) is very small for the given order, '
                'better use a larger scale!')

        # Create t vector which indicates the x-position
    t = np.arange(-N / 2.0 + 0.5, N / 2.0, 1.0, dtype=np.float64)

    # Get kernel
    k = _gaussian_kernel(sigma, order, t)

    # Done
    if returnt:
        return k, t
    else:
        return k


def gaussian_kernel2D(mu, cov, samples):
    """ gaussian_kernel2D(mu, cov, samples)
    2D Gaussian kernel with the specified means and covariance
    """

    twod_gaussian = np.random.multivariate_normal(mu, cov, samples)

    return twod_gaussian



def smooth_signal(signal, kernel):
    """ smooth_signal(signal, kernel)
    Smooth the given 1D signal by convolution with a specified kernel
    """
    return np.convolve(signal, kernel, mode='same')


def compute_curvature(curve, sigma):
    """ compute_curvature(curve, sigma)
    Compute the curvature of a 2D curve as given in Mohkatarian et. al.
    and return the curvature signal at the given sigma

    Components of the 2D curve are:
    curve[0,:] and curve[1,:]

    Parameters
    -------------
    curve : numpy matrix
        Two row matrix representing 2D curve
    sigma : float
        Kernel width

    """

    if curve[0, :].size < 2:
        raise Exception("Curve must have at least 2 points")

    sigx = curve[0, :]
    sigy = curve[1, :]
    g = gaussian_kernel(sigma, 0, sigx.size, False)
    g_s = gaussian_kernel(sigma, 1, sigx.size, False)
    g_ss = gaussian_kernel(sigma, 2, sigx.size, False)

    X_s = smooth_signal(sigx, g_s)
    Y_s = smooth_signal(sigy, g_s)
    X_ss = smooth_signal(sigx, g_ss)
    Y_ss = smooth_signal(sigy, g_ss)

    kappa = ((X_s * Y_ss) - (X_ss * Y_s)) / (X_s**2 + Y_s**2)**(1.5)

    return kappa, smooth_signal(sigx, g), smooth_signal(sigy, g)


def rebin(a, shape):
    """ rebin a piece of data """
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_and_pack_data(data, lenx, leny):
    """ load_and_pack_data(data, lenx, leny)
    Given a dictionary of training data with {data :label},
    loads the data into a pair X,Y
    """

    Xdata = np.zeros(shape=(len(data), (lenx * leny)))
    Ydata = np.zeros(len(data))

    idx = 0
    for f, v in data.items():
        print f, v, idx
        d = np.load(f)
        Xdata[idx, :] = rebin(d, [lenx, leny]).reshape(lenx * leny)
        Ydata[idx] = v
        idx += 1
    return Xdata, Ydata
