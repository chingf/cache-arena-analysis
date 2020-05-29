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
    fr = np.convolve(spikes, smoothing_kernel, "valid")
    fr_frames = np.arange(spikes.size)[
        smoothing_kernel.size//2:-smoothing_kernel.size//2+1
        ]
    return fr, fr_frames

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

def circular_shuffle(spikes):
    spikes = spikes.copy()
    shift = np.random.choice(np.arange(1, spikes.size))
    return np.roll(spikes, shift)

def get_cache_frames(
    window, wedges, wedge_frames,
    cache_site, cache_sites,
    cache_frames_poke, cache_frames_enter, cache_frames_exit
    ):
    
    event_idxs = np.argwhere(cache_sites == cache_site).squeeze()
    cache_pokes = cache_frames_poke[event_idxs]
    cache_frames_enter = cache_frames_enter[event_idxs]
    cache_frames_exit = cache_frames_exit[event_idxs]
    wedge_frames = wedge_frames[wedges == cache_site]
    if event_idxs.size == 1:
        cache_pokes = [cache_pokes]
        cache_frames_enter = [cache_frames_enter]
        cache_frames_exit = [cache_frames_exit]
    cache_frames = [np.arange(c-window, c+window+1) for c in cache_pokes]
    cache_frames = np.concatenate(cache_frames)
    visit_frames = [
        np.arange(c, cache_frames_exit[i]) for i, c in enumerate(cache_frames_enter)
        ]
    visit_frames = np.concatenate(visit_frames)
    noncache = np.logical_not(np.isin(wedge_frames, cache_frames))
    nonvisit = np.logical_not(np.isin(wedge_frames, visit_frames))
    noncache_frames = wedge_frames[np.logical_and(noncache, nonvisit)]
    return cache_frames.astype(int), noncache_frames.astype(int)

def get_visit_frames(
    window, wedges, wedge_frames,
    cache_site, cache_sites,
    cache_frames_poke, cache_frames_enter, cache_frames_exit
    ):
    
    event_idxs = np.argwhere(cache_sites == cache_site).squeeze()
    cache_frames_enter = cache_frames_enter[event_idxs]
    cache_frames_exit = cache_frames_exit[event_idxs]
    cache_frames_poke = cache_frames_poke[event_idxs]
    wedge_frames = wedge_frames[wedges == cache_site]
    if event_idxs.size == 1:
        cache_frames_enter = [cache_frames_enter]
        cache_frames_exit = [cache_frames_exit]
        cache_frames_poke = [cache_frames_poke]
    visit_frames = [
        np.arange(enter, cache_frames_exit[i] + 1) for i, enter in enumerate(cache_frames_enter)
        ]
    visit_frames = np.concatenate(visit_frames)
    window_frames = []
    for i, enter in enumerate(cache_frames_enter):
        poke = cache_frames_poke[i]
        exit = cache_frames_exit[i]
        prepoke_time = poke - enter
        postpoke_time = exit - poke
        total_time = exit - enter
        if prepoke_time >= window and postpoke_time >= window:
            _window_frames = np.arange(poke-window, poke+window+1)
        elif prepoke_time < 30 and total_time > (window*2 + 1):
            _window_frames = np.arange(enter, enter+window*2+1)
        elif postpoke_time < 30 and total_time > (window*2 + 1):
            _window_frames = np.arange(exit-window*2, exit+1)
        else:
            _window_frames = np.arange(enter, exit+1)
        window_frames.append(_window_frames)
    if len(window_frames) == 0:
        return np.array([]), np.array([])
    window_frames = np.concatenate(window_frames)
    nonvisit = np.logical_not(np.isin(wedge_frames, visit_frames))
    nonvisit_frames = wedge_frames[nonvisit]
    return window_frames, nonvisit_frames

def get_general_cache_frames(
    window, wedges, wedge_frames,
    cache_sites,
    cache_frames_poke, cache_frames_enter, cache_frames_exit
    ):

    cache_frames = []
    noncache_frames = []
    for cache_site in np.unique(cache_sites):
        _cache_frames, _noncache_frames = get_cache_frames(
            window, wedges, wedge_frames,
            cache_site, cache_sites,
            cache_frames_poke, cache_frames_enter, cache_frames_exit
            )
        cache_frames.append(_cache_frames)
        noncache_frames.append(_noncache_frames)
    cache_frames = np.concatenate(cache_frames)
    noncache_frames = np.concatenate(noncache_frames)
    overlapping_frames = np.isin(noncache_frames, cache_frames)
    noncache_frames = noncache_frames[np.logical_not(overlapping_frames)]
    return cache_frames, noncache_frames

def get_general_visit_frames(
    window, wedges, wedge_frames,
    cache_sites,
    cache_frames_poke, cache_frames_enter, cache_frames_exit
    ):
    
    cache_frames = []
    noncache_frames = []
    for cache_site in np.unique(cache_sites):
        _cache_frames, _noncache_frames = get_visit_frames(
            window, wedges, wedge_frames,
            cache_site, cache_sites,
            cache_frames_poke, cache_frames_enter, cache_frames_exit
            )
        cache_frames.append(_cache_frames)
        noncache_frames.append(_noncache_frames)
    cache_frames = np.concatenate(cache_frames)
    noncache_frames = np.concatenate(noncache_frames)
    overlapping_frames = np.isin(noncache_frames, cache_frames)
    noncache_frames = noncache_frames[np.logical_not(overlapping_frames)]
    return cache_frames, noncache_frames
