import numpy as np
from scipy.stats import binned_statistic_2d

fps = 20

def get_velocity(f):
    x, y, frames = get_xy(f, in_bound=False)
    delta_x = x[1:] - x[:-1]
    delta_y = y[1:] - y[:-1]
    frames = frames[1:]
    velocity = np.sqrt(np.square(delta_x) + np.square(delta_y)) # pixels/frame
    velocity = velocity*fps # pixels/s
    smoothing_kernel = np.ones(fps)/fps
    velocity = np.convolve(velocity, smoothing_kernel, "valid")
    frames = frames[:velocity.size]
    return velocity, frames

def get_fr(spikes, window=fps):
    """
    Returns boxcar smoothing of spikes into firing rates.

    Args:
        spikes: either (frames,) or (neurs, frames)
    Returns:
        array of same shape as spikes, with firing rate calculated over frames
    """

    smoothing_kernel = np.ones(window+1)/(window+1) # One sec smoothing
    if len(spikes.shape) == 2:
        fr = []
        for neur in range(spikes.shape[0]):
            fr.append(np.convolve(spikes[neur], smoothing_kernel, "same"))
        fr = np.array(fr)
    else:
        fr = np.convolve(spikes, smoothing_kernel, "same")
    return fr

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
#     if np.sum(np.isnan(dist)) > 0:
#         import pdb; pdb.set_trace()
    return dist
