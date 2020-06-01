import os
import h5py
import pickle
import numpy as np
import pandas as pd
import pdb
from scipy.io import loadmat
from scipy import optimize
from math import pi, log2

mat_files = [m for m in os.listdir("data") if m.endswith('mat')]
session_list = pd.read_excel('data/CacheRetrieveSessionList.xlsx', index_col=0)
fps = 20

def estimate_center(x, y):
    method_2 = "leastsq"

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    center_2, ier = optimize.leastsq(f_2, center_estimate)
    return center_2

def get_xy(f, in_bound=True):
    x = np.squeeze(np.array(f['X']))
    y = np.squeeze(np.array(f['Y']))
    x_c, y_c = estimate_center(x, y)
    x -= x_c; y -= y_c
    length = np.sqrt(np.square(x) + np.square(y))
    frames = np.arange(x.size)
    if in_bound:
        oob = np.logical_or(length <= 145, length >= 215)
        x = x[np.logical_not(oob)]
        y = y[np.logical_not(oob)]
        frames = frames[np.logical_not(oob)]
    return x, y, frames

def get_theta(f):
    x, y, frames = get_xy(f, in_bound=True)
    theta = np.arctan2(y, x)
    theta = np.mod(theta, 2*pi)
    boundaries = np.linspace(0, 2*pi, 16, endpoint=False)
    boundaries = np.append(boundaries, [2*pi])
    theta = np.digitize(theta, boundaries)
    return theta, frames

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

def get_wedges(f):
    x, y, frames = get_xy(f, in_bound=True)
    theta = np.mod(np.arctan2(y, x), 2*pi)
    boundaries = np.linspace(0, 2*pi, 16, endpoint=False)
    boundaries = np.append(boundaries, [2*pi])
    wedges = np.digitize(theta, boundaries)
    wedges = np.mod(16-wedges, 16) + 1
    return wedges, frames

def get_fr(spikes):
    smoothing_kernel = np.ones(fps+1)/(fps+1) # One sec smoothing
    if len(spikes.shape) == 2:
        fr = []
        for neur in range(spikes.shape[1]):
            fr.append(np.convolve(spikes[:,neur], smoothing_kernel, "same"))
        fr = np.array(fr).T
    else:
        fr = np.convolve(spikes, smoothing_kernel, "same")
    return fr

def get_mutual_info(contexts, fr):
    mean_fr = np.mean(fr)
    mutual_info = 0
    for ctxt in np.unique(contexts):
        prob = np.sum(contexts==ctxt)/contexts.size
        ctxt_mean_fr = np.mean(fr[contexts==ctxt])
        try:
            log_term = log2(ctxt_mean_fr/mean_fr)
        except:
            log_term = 0
        if np.isnan(log_term):
            log_term = 0
        mutual_info += prob*ctxt_mean_fr*log_term
    return mutual_info

def get_mutual_info_mat(contexts, fr):
    mean_fr = np.mean(fr, axis=0)
    mutual_info = np.zeros(mean_fr.shape)
    for ctxt in np.unique(contexts):
        prob = np.sum(contexts==ctxt)/contexts.size
        ctxt_mean_fr = np.mean(fr[contexts==ctxt,:], axis=0)
        log_term = np.log2(ctxt_mean_fr/mean_fr)
        log_term[np.isnan(log_term)] = 0
        log_term[np.isinf(log_term)] = 0
        mutual_info += prob*ctxt_mean_fr*log_term
    return mutual_info

def get_cache_index(
    cache_frames, noncache_frames, neur_fr, percentile
    ):
    noncache_frs = []
    cache_fr = np.mean(neur_fr[cache_frames])
    for nc in noncache_frames:
        if nc.size < cache_frames.size:
            continue
        noncache_fr = np.sort(neur_fr[nc])
        noncache_fr = noncache_fr[int(noncache_fr.size*percentile):]
        noncache_fr = np.mean(noncache_fr)
        noncache_frs.append(noncache_fr)
    return np.sum(noncache_frs < cache_fr)/len(noncache_frs)

def shuffle_two_nparrays(a1, a2):
    all_vals = np.concatenate([a1.flatten(), a2.flatten()])
    for _ in range(4):
        np.random.shuffle(all_vals)
    shuff_a1 = all_vals[:a1.size].reshape(a1.shape)
    shuff_a2 = all_vals[a1.size:].reshape(a2.shape)
    return shuff_a1, shuff_a2

def shuffle_1darray_lists(a, l):
    l_vals = np.concatenate(l)
    all_vals = np.concatenate([a, l_vals])
    for _ in range(4):
        np.random.shuffle(all_vals)
    shuff_a = all_vals[:a.size]
    shuff_l = []
    idx = a.size
    for _l in l:
        shuff_l.append(all_vals[idx : idx+_l.size])
        idx += _l.size
    return shuff_a, shuff_l

def circular_shuffle(spikes):
    spikes = spikes.copy()
    shift = np.random.choice(np.arange(1, spikes.size))
    if len(spikes.shape) == 2:
        for neur in range(spikes.shape[1]):
            shift = np.random.choice(np.arange(1, spikes.size))
            spikes[:,neur] = np.roll(spikes[:,neur], shift)
        return spikes
    else:
        return np.roll(spikes, shift)

def get_poke_centered_cache(poke, enter, exit, window):
    total_window_len = window*2 + 1
    prepoke_time = poke - enter
    postpoke_time = exit-poke
    total_time = exit-enter
    if prepoke_time >= window and postpoke_time >= window:
        cache_frames = np.arange(
            poke-window, poke + window+1
            )
    elif prepoke_time < window and total_time > (total_window_len):
        cache_frames = np.arange(
            enter, enter + total_window_len
            )
    elif postpoke_time < window and total_time > total_window_len:
        cache_frames = np.arange(
            exit - total_window_len - 1, exit + 1
            )
    else:
        cache_frames = np.arange(enter, exit + 1)
    return cache_frames

def get_poke_centered_frames(
    window, wedges,
    cache_site, cache_sites,
    cache_frames_poke, cache_frames_enter, cache_frames_exit
    ):
    """ Returns a list of cache_frames and a list of noncache frames"""
    
    total_window_len = window*2 + 1
    cache_frames = []
    noncache_frames = []
    event_idxs = np.argwhere(cache_sites == cache_site).reshape((-1,1))
    # Collect the caching-related frames
    for event_idx in event_idxs:
        poke = cache_frames_poke[event_idx]
        enter = cache_frames_enter[event_idx]
        exit = cache_frames_exit[event_idx]
        _cache_frames = get_poke_centered_cache(poke, enter, exit, window)
        cache_frames.append(_cache_frames)
    cache_frames = np.concatenate(cache_frames)
    # Collect the noncaching-related frames
    visit_frames = [np.arange(
        cache_frames_enter[i], cache_frames_exit[i] + 1
        ) for i in event_idxs]
    visit_frames = np.concatenate(visit_frames)
    wedge_frames = np.argwhere(wedges==cache_site).reshape((-1,1))
    noncache_frames = wedge_frames[
        np.logical_not(np.isin(wedge_frames, visit_frames))
        ]
    return cache_frames, noncache_frames

def get_hop_centered_cache(enter, exit, window):
    return np.arange(enter - window, enter + window + 1)

def get_hop_centered_frames(
    window, wedges,
    cache_site, cache_sites,
    cache_frames_enter, cache_frames_exit
    ):
    """ Returns a list of cache_frames and a list of noncache frames"""
    
    total_window_len = window*2 + 1
    cache_frames = []
    noncache_frames = []
    event_idxs = np.argwhere(cache_sites == cache_site).reshape((-1,1))
    # Collect the caching-related frames
    for event_idx in event_idxs:
        enter = cache_frames_enter[event_idx]
        exit = cache_frames_exit[event_idx]
        _cache_frames = get_hop_centered_cache(enter, exit, window)
        cache_frames.append(_cache_frames)
        if _cache_frames.size != total_window_len:
            import pdb; pdb.set_trace()
    cache_frames = np.concatenate(cache_frames)
    # Collect the noncaching-related frames
    visit_frames = [np.arange(
        cache_frames_enter[i] - window, cache_frames_enter[i] + window + 1
        ) for i in event_idxs]
    visit_frames = np.concatenate(visit_frames)
    wedge_frames = np.argwhere(wedges==cache_site).reshape((-1,1))
    wedge_frames = np.split(
        wedge_frames, np.where(np.diff(wedge_frames) != 1)[0]+1
        )
    noncache_frames = []
    for wedge_frame in wedge_frames:
        enter = wedge_frame[0]
        if enter < window:
            padding = total_window_len - (enter+window+1)
            noncache_window = np.concatenate([
                np.ones(padding)*-1, np.arange(0, enter+window+1)
                ])
        elif exit > (wedges.size - window):
            padding = total_window_len - (wedges.size - (enter-window))
            noncache_window = np.concatenate([
                np.arange(enter-window, wedges.size), np.ones(padding)*-1
                ])
        else:
            noncache_window = np.arange(enter-window, enter+window+1)
        if noncache_window.size != total_window_len:
            import pdb; pdb.set_trace()
        noncache_frames.append(noncache_window)
    noncache_frames = np.array(noncache_frames)
    noncache_frames[np.isin(noncache_frames, visit_frames)] = -1
    return cache_frames, noncache_frames

def get_cache_index_frames(
    wedges, cache_sites, pokes, enters, exits,
    fr, window
    ):
    all_noncache_frames = {}
    for cache_site in np.unique(cache_sites):
        _, noncache_frames = get_poke_centered_frames(
            window, wedges, cache_site, cache_sites,
            pokes, enters, exits
            )
        noncache_frames = np.split(
            noncache_frames, np.where(np.diff(noncache_frames) != 1)[0]+1
            )
        all_noncache_frames[cache_site] = noncache_frames
    all_cache_frames = {}
    for idx in range(cache_sites.size):
        cache_frames = get_poke_centered_cache(
            pokes[idx], enters[idx], exits[idx],
            window
            )
        all_cache_frames[idx] = cache_frames
    return all_cache_frames, all_noncache_frames

def calc_cache_index(fr, cache_frames, noncache_frames):
    cache_frs = np.mean(fr[cache_frames,:], axis=0)
    noncache_frs = []# (Noncache visits, neurs) array
    for noncache_frame in noncache_frames:
        noncache_fr = np.mean(fr[noncache_frame,:], axis=0)
        noncache_frs.append(noncache_fr)
    noncache_frs = np.array(noncache_frs)
    cache_idx = np.sum(noncache_frs < cache_frs, axis=0)
    cache_idx = cache_idx/noncache_frs.shape[0]
    return cache_idx
