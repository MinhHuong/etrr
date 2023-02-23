# data manipulation
import numpy as np

# constants
atts = ['PPOR', 'TPOR', 'IPOR', 'FDCV', 'FDCF']


def compute_profile(cycles_):
    """
    Compute the common profile from a set of signals

    Return: an array of shape (max_step, dim)
    """
    max_len = max([len(cyc) for cyc in cycles_])
    dim = len(atts)
    common_mean = np.zeros((max_len, dim))
    for t in range(max_len):
        for j in range(dim):
            common_mean[t, j] = np.mean([cyc[t, j] for cyc in cycles_ if len(cyc) > t])
    return common_mean


def compute_envelope(cycles_):
    """
    Compute the envelope from a set of multivariate signals (very simple)

    Parameters
    ----------
    cycles_
        A set of signals

    Returns
    -------
    set of max values, set of min values
    """
    dim = len(atts)
    max_len = max([len(s) for s in cycles_])
    max_points = np.zeros((max_len, dim))
    min_points = np.zeros((max_len, dim))
    # find the envelope
    for t in range(max_len):
        for j in range(dim):
            vals = [s[t, j] for s in cycles_ if len(s) > t]
            max_points[t, j] = max(vals)
            min_points[t, j] = min(vals)
    return max_points, min_points


def compute_area(cycle_, common_):
    """
    Compute the area between two curves

    Parameters
    ----------
    cycle_
    common_

    Returns
    -------

    """
    dim = len(atts)
    areas_ = 0
    min_len = min(len(cycle_), len(common_))
    for j in range(dim):
        region = np.trapz(np.abs(cycle_[:min_len, j] - common_[:min_len, j]))
        areas_ += region
    areas_ /= dim
    return areas_
