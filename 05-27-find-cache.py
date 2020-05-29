import os
import h5py
import pickle
import numpy as np
import pandas as pd
import pdb
from scipy.io import loadmat
from scipy import optimize
from math import pi, log2
from utils import *
from multiprocessing import Pool

mat_files = [m for m in os.listdir("data") if m.endswith('mat')]
session_list = pd.read_excel('data/CacheRetrieveSessionList.xlsx', index_col=0)
fps = 20

def pool_func(outer_func, find_func, window, shuffle, key_name):
    val = outer_func(find_func, window, shuffle)
    return key_name, val

def find_specific_cache(find_func, window, shuffle):
    results = {}
    for mat_file in mat_files:
        results[mat_file] = {}
        f = h5py.File("data/" + mat_file, 'r')
        _, wedge_frames = get_wedges(f)
        wedges = np.array(f['whichWedge']).squeeze()
        wedges = wedges[np.isin(np.arange(wedges.size), wedge_frames)]
        cache_sites = np.array(f['CacheSites']).squeeze()
        cache_frames_poke = np.array(f['CacheFrames']).squeeze().astype(int) - 1
        cache_frames_enter = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
        cache_frames_exit = np.array(f['CacheFramesExit']).squeeze().astype(int) - 1
        was_retrieval = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
        was_cache = np.logical_not(was_retrieval)
        spikes = np.array(f['S'])
        for cache_site in np.unique(cache_sites):
            results[mat_file][cache_site] = []
            cache_frames, noncache_frames = find_func(
                window, wedges, wedge_frames,
                cache_site, cache_sites,
                cache_frames_poke, cache_frames_enter, cache_frames_exit
                )
            if cache_frames.size == 0:
                results[mat_file][cache_site].append(np.nan)
                continue 
            for neur in np.arange(spikes.shape[1]):
                neur_spikes = spikes[:, neur]
                neur_fr, fr_frames = get_fr(neur_spikes)
                cf = np.intersect1d(fr_frames, cache_frames)
                ncf = np.intersect1d(fr_frames, noncache_frames)
                combined_frames = np.concatenate([cf, ncf])
                contexts = np.concatenate([np.ones(cf.size), np.zeros(ncf.size)])
                if shuffle:
                    for _ in range(5):
                        np.random.shuffle(contexts)
                cache_info = get_mutual_info(
                    contexts, neur_fr[np.isin(fr_frames, combined_frames)]
                    )
                shuffled_info = []
                shuffled_peak_fr = []
                for _ in range(110):
                    shuffled_spikes = circular_shuffle(neur_spikes)
                    shuffled_fr, _ = get_fr(shuffled_spikes)
                    shuffled_info.append(get_mutual_info(
                        contexts, shuffled_fr[np.isin(fr_frames, combined_frames)]
                        ))
                    shuffled_peak_fr.append(shuffled_fr.max())
                shuffled_info = np.array(shuffled_info)
                high_mutual_info = np.sum(shuffled_info < cache_info) > 0.99*shuffled_info.size
                if high_mutual_info:
                    results[mat_file][cache_site].append(neur)
    return results

def find_general_cache(find_func, window, shuffle):
    results = {}
    for mat_file in mat_files:
        # Load data
        results[mat_file] = []
        f = h5py.File("data/" + mat_file, 'r')
        _, wedge_frames = get_wedges(f)
        wedges = np.array(f['whichWedge']).squeeze()
        wedges = wedges[np.isin(np.arange(wedges.size), wedge_frames)]
        cache_sites = np.array(f['CacheSites']).squeeze()
        cache_frames_poke = np.array(f['CacheFrames']).squeeze().astype(int) - 1
        cache_frames_enter = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
        cache_frames_exit = np.array(f['CacheFramesExit']).squeeze().astype(int) - 1
        was_retrieval = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
        was_cache = np.logical_not(was_retrieval)
        spikes = np.array(f['S'])
        cache_frames, noncache_frames = find_func(
            window, wedges, wedge_frames, cache_sites,
            cache_frames_poke, cache_frames_enter, cache_frames_exit
            )
        for neur in np.arange(spikes.shape[1]):
            neur_spikes = spikes[:, neur]
            neur_fr, fr_frames = get_fr(neur_spikes)
            cf = np.intersect1d(fr_frames, cache_frames)
            ncf = np.intersect1d(fr_frames, noncache_frames)
            combined_frames = np.concatenate([cf, ncf])
            contexts = np.concatenate([np.ones(cf.size), np.zeros(ncf.size)])
            if shuffle:
                for _ in range(3):
                    np.random.shuffle(contexts)
            cache_info = get_mutual_info(
                contexts, neur_fr[np.isin(fr_frames, combined_frames)]
                )
            shuffled_info = []
            shuffled_peak_fr = []
            for _ in range(110):
                shuffled_spikes = circular_shuffle(neur_spikes)
                shuffled_fr, _ = get_fr(shuffled_spikes)
                shuffled_info.append(get_mutual_info(
                    contexts, shuffled_fr[np.isin(fr_frames, combined_frames)]
                    ))
                shuffled_peak_fr.append(shuffled_fr.max())
            shuffled_info = np.array(shuffled_info)
            high_mutual_info = np.sum(shuffled_info < cache_info) > 0.99*shuffled_info.size
            if high_mutual_info:
                results[mat_file].append(neur)
    return results

def run_specific_cache():
    #pickle_names = ["specific-cache", "specific-visit"]
    #find_funcs = [get_cache_frames, get_visit_frames]
    #windows = [20, 30, 40]
    pickle_names = ["specific-cache"]
    find_funcs = [get_cache_frames]
    windows = [30, 40]
    for idx, pickle_name in enumerate(pickle_names):
        find_func = find_funcs[idx]
        for window in windows:
            results = {}
            key_names = ["original", "shuffled1", "shuffled2", "shuffled3"]
            shuffles = [False, True, True, True]
            args = [(
                find_specific_cache, find_func, window, shuffles[i], key_names[i]
                ) for i in range(len(shuffles))]
            pool = Pool(processes=4)
            pool_results = pool.starmap(pool_func, args)
            pool.close()
            pool.join()
            for pool_result in pool_results:
                key, val = pool_result
                results[key] = val
            with open(pickle_name + "-" + str(window) + ".p", "wb") as f:
                pickle.dump(results, f)

def run_general_cache():
    pickle_names = ["general-cache", "general-visit"]
    find_funcs = [get_general_cache_frames, get_general_visit_frames]
    windows = [20, 30, 40]
    for idx, pickle_name in enumerate(pickle_names):
        find_func = find_funcs[idx]
        for window in windows:
            results = {}
            key_names = ["original", "shuffled1", "shuffled2", "shuffled3"]
            shuffles = [False, True, True, True]
            args = [(
                find_general_cache, find_func, window, shuffles[i], key_names[i]
                ) for i in range(len(shuffles))]
            pool = Pool(processes=4)
            pool_results = pool.starmap(pool_func, args)
            pool.close()
            pool.join()
            for pool_result in pool_results:
                key, val = pool_result
                results[key] = val
            with open(pickle_name + "-" + str(window) + ".p", "wb") as f:
                pickle.dump(results, f)

def find_place():
    results = {}
    for mat_file in mat_files:
        # Load data
        results[mat_file] = []
        f = h5py.File("data/" + mat_file, 'r')
        _, wedge_frames = get_wedges(f)
        wedges = np.array(f['whichWedge']).squeeze()
        wedges = wedges[np.isin(np.arange(wedges.size), wedge_frames)]
        spikes = np.array(f['S'])
        
        for neur in np.arange(spikes.shape[1]):
            neur_spikes = spikes[:, neur]
            neur_fr, fr_frames = get_fr(neur_spikes)
            valid_frames = np.intersect1d(fr_frames, wedge_frames)
            spatial_info = get_mutual_info(
                wedges[np.isin(wedge_frames, valid_frames)],
                neur_fr[np.isin(fr_frames, valid_frames)]
                )
            shuffled_info = []
            shuffled_peak_fr = []
            for _ in range(110):
                shuffled_spikes = circular_shuffle(neur_spikes)
                shuffled_fr, _ = get_fr(shuffled_spikes)
                shuffled_info.append(get_mutual_info(
                    wedges[np.isin(wedge_frames, valid_frames)],
                    shuffled_fr[np.isin(fr_frames, valid_frames)]
                    ))
                shuffled_peak_fr.append(shuffled_fr.max())
            shuffled_info = np.array(shuffled_info)
            high_mutual_info = np.sum(shuffled_info < spatial_info) > 0.99*shuffled_info.size
            if high_mutual_info:
                results[mat_file].append(neur)
    with open("spatial.p", "wb") as f:
        pickle.dump(results, f)

run_specific_cache()
#run_general_cache()
#find_place()
