# Gaussian gerivatives using Hermite polynomials

import numpy as np


def _gaussian_kernel(sigma, order, t):
    """ _gaussian_kernel(sigma, order, t)
    Calculate a Gaussian kernel of the given sigma and with the given
    order, using the given t-values.
    """

    # if sigma 0, kernel is a single 1.
    if sigma == 0:
        return np.array([1.0])

    # precalculate some stuff
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
    2D gaussian kernel with the soecified means and covariances
    """

    g2 = np.random.multivariate_normal(mu, cov, samples)

    return g2
