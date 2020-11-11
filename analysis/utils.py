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

def in_ellipse(points_x, points_y, ellipse_center, ellipse_semiaxis):
    """
    For an array of points in XY space, returns whether the points are in the
    defined ellipse.
    """

    x_length = (np.square(points_x - ellipse_center[0])/(ellipse_semiaxis[0]**2))
    y_length = (np.square(points_y - ellipse_center[1])/(ellipse_semiaxis[1]**2))
    return (x_length + y_length) <= 1

def get_max_consecutive(arr):
    """ Returns the maximum length of consecutive True runs in the array. """

    if np.sum(arr) == arr.size: return arr.size
    max_consecutive = 0
    curr_consecutive = 0
    in_consecutive_run = False
    for i in np.arange(arr.size):
        if arr[i] and not in_consecutive_run: # Start of a run
            in_consecutive_run = True
            curr_consecutive = 1
        elif arr[i] and in_consecutive_run: # Middle of a run
            curr_consecutive += 1
        elif not arr[i] and in_consecutive_run: # End of a run
            if curr_consecutive > max_consecutive:
                max_consecutive = curr_consecutive
            in_consecutive_run = False
            curr_consecutive = 0
        else:
            continue
    return max_consecutive

