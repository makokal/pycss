

from utils import gaussian_kernel, compute_curvature
import numpy as np


class CurvatureScaleSpace(object):
    """ Curvature Scale Space

    A simple curvature scale space implementation based on
    Mohkatarian et. al. paper. Full algorithm detailed in
    Okal msc thesis

    """

    def __init__(self):
        pass

    def find_zero_crossings(self, kappa):
        """ find_zero_crossings(kappa)
        Locate the zero crossing points of the curvature signal kappa(t)
        """

        crossings = []

        for i in range(0, kappa.size - 2):
            if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
                crossings.append(i)

        return crossings

    def generate_css(self, curve, max_sigma, step_sigma):
        """ generate_css(curve, max_sigma, step_sigma)
        Generates a CSS image representation by repetatively smoothing the initial curve L_0 with increasing sigma
        """

        cols = curve[0, :].size
        rows = max_sigma // step_sigma
        css = np.zeros(shape=(rows, cols))

        srange = np.linspace(1, max_sigma - 1, rows)
        for i, sigma in enumerate(srange):
            kappa, sx, sy = compute_curvature(curve, sigma)

            # find interest points
            xs = self.find_zero_crossings(kappa)

            # save the interest points
            if len(xs) > 0 and sigma < max_sigma - 1:
                for c in xs:
                    css[i, c] = sigma  # change to any positive

            else:
                return css

    def generate_visual_css(self, rawcss, closeness, return_all=False):
        """ generate_visual_css(rawcss, closeness)
        Generate a 1D signal that can be plotted to depict the CSS by taking
        column maximums. Further checks for close interest points and nicely
        smoothes them with weighted moving average
        """

        flat_signal = np.amax(rawcss, axis=0)

        # minor smoothing via moving averages
        window = closeness
        weights = gaussian_kernel(window, 0, window, False)  # gaussian weights
        sig = np.convolve(flat_signal, weights)[window - 1:-(window - 1)]

        maxs = []

        # get maximas
        w = sig.size

        for i in range(1, w - 1):
            if sig[i - 1] < sig[i] and sig[i] > sig[i + 1]:
                maxs.append([i, sig[i]])

        if return_all:
            return sig, maxs
        else:
            return sig

    def generate_eigen_css(self, rawcss, return_all=False):
        """ generate_eigen_css(rawcss, return_all)
        Generates Eigen-CSS features
        """
        rowsum = np.sum(rawcss, axis=0)
        csum = np.sum(rawcss, axis=1)

        # hack to trim c
        colsum = csum[0:rowsum.size]

        freq = np.fft.fft(rowsum)
        mag = abs(freq)

        tilde_rowsum = np.fft.ifft(mag)

        feature = np.concatenate([tilde_rowsum, colsum], axis=0)

        if not return_all:
            return feature
        else:
            return feature, rowsum, tilde_rowsum, colsum


class SlicedCurvatureScaleSpace(CurvatureScaleSpace):
    """ Sliced Curvature Scale Space

    A implementation of the SCSS algorithm as detailed in Okal thesis

    """
    def __init__(self):
        pass

    def generate_scss(self, curves, resample_size, max_sigma, step_sigma):
        """ generate_scss
        Generate the SCSS image
        """

        scss = np.zeros(shape=(len(curves), resample_size))  # TODO - fix this hack
        # maxs = np.zeros(shape=(len(curves), resample_size))

        for i, curve in enumerate(curves):
            scss[i, :] = self.generate_css(curve, max_sigma, step_sigma)

        return scss
