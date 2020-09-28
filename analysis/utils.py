import os
import h5py
import pickle
import numpy as np
import pandas as pd
import pdb
from scipy.io import loadmat
from scipy import optimize
from math import pi, log2

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
    smoothing_kernel = np.ones(window+1)/(window+1) # One sec smoothing
    if len(spikes.shape) == 2:
        fr = []
        for neur in range(spikes.shape[0]):
            fr.append(np.convolve(spikes[neur], smoothing_kernel, "same"))
        fr = np.array(fr)
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

def circular_shuffle_1darray_lists(a, l):
    l_vals = np.concatenate(l)
    all_vals = np.concatenate([a, l_vals])
    shift = np.random.choice(np.arange(1, all_vals.size))
    all_vals = np.roll(all_vals, shift)
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

def circular_shuffle_two_nparrays(a1, a2):
    all_vals = np.concatenate([a1.flatten(), a2.flatten()])
    shift = np.random.choice(np.arange(1, all_vals.size))
    all_vals = np.roll(all_vals, shift)
    shuff_a1 = all_vals[:a1.size].reshape(a1.shape)
    shuff_a2 = all_vals[a1.size:].reshape(a2.shape)
    return shuff_a1, shuff_a2

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

