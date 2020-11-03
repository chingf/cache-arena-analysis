import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter1d

fps = 20

def cart2pol(x, y):
    rhos = np.sqrt(np.square(x) + np.square(y))
    thetas = np.arctan2(y, x)
    return(rhos, thetas)

def get_fr(spikes, window=fps):
    """
    Returns boxcar smoothing of spikes into firing rates.

    Args:
        spikes: either (frames,) or (neurs, frames)
    Returns:
        array of same shape as spikes, with firing rate calculated over frames
    """

    if len(spikes.shape) == 2:
        return gaussian_filter1d(spikes, sigma=window, axis=1)
    else:
        return gaussian_filter1d(spikes, sigma=window)

def gen_2d_bins(x, y, n):
    """ Given a series of (x,y) coordinates, bins into a n x n grid """

    _, x_edge, y_edge, binnumber_2d = binned_statistic_2d(
        x, y, x, 'count', bins=np.linspace(0, 430, num=n, endpoint=True),
        expand_binnumbers=True
        )
    binnumber_lin = (binnumber_2d[0]*n) + binnumber_2d[1]
    return binnumber_2d, binnumber_lin

def popln_overlap(A, b):
    """
    Returns population overlap as defined by X
    
    Args:
        A: ()
    """

    A = A.T
    adotb = np.dot(A, b)
    asumsq = np.sum(np.square(A), axis=1)
    bsumsq = np.sum(np.square(b))
    dist = adotb/(asumsq + bsumsq - adotb)
    return dist
