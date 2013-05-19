
import numpy as np
from scipy.signal import resample


def _clength(curve):
    r, c = curve.shape
    ln = 0
    for i in range(c-1):
        ln += np.linalg.norm(curve[:, i]-curve[:, i+1], ord=2)
    return ln


def resample_curve(curve, samples, thresh= 0.01, closed=False):
    """ resample_curve(curve, samples, thresh, closed)
    Resample a simple 2D curve with equidistant points
    """

    if curve[0,:].size < 2 or samples < 2:
        raise Exception("Curve must have at least 2 points, and samples > 1")

    curve_length = _clength(curve)
    resample_length = curve_length / samples

    rcurve = np.zeros(shape=(2, samples))
    rcurve[:, 0] = curve[:, 0]

    dist = 0.0
    index = 0

    for i in range(1, samples):
        # assert(index < curve[0,:].size - 1), "Index out of range(beyond curve)"

        if index < curve[0,:].size-1:

            last_dist = np.linalg.norm(curve[:, index]-curve[:, index+1], ord=2)
            dist += last_dist

            if dist >= resample_length:
                point_dist = last_dist - (dist - resample_length)
                p_new = curve[:, index+1] - curve[:, index]
                p_new /= np.linalg.norm(p_new, ord=2)

                # assign point to rcurve
                assert(i < samples-1), "Goint out or array"
                rcurve[:, i] = curve[:, index] + (p_new * point_dist)

                # update distances
                dist = last_dist - point_dist
                i += 1

                # check any further point insertion
                while (dist - resample_length) > thresh:
                    assert(i < samples-1), "Goint out or array"
                    rcurve[:, i] = rcurve[:, i-1] + (p_new * resample_length)
                    dist -= resample_length
                    i -= 1
                # endwhile
            # endif
            index += 1
        # endif
    # endfor

    return rcurve


def resample_via_fft(curve, samples):
    """ resample_via_fft(curve, samples)
    Resample the curve using scily signal processing utility via Fourier and zero padding
    """

    rx = resample(curve[0,:], samples)
    ry = resample(curve[1,:], samples)

    # rsig = resample(curve, samples, axis=0)

    return np.array([rx, ry])
    # return rsig

    
