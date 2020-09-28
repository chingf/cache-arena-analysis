import numpy as np
from analysis.utils import *

class ExpData(object):
    """
    Given data in Emily's file format, will extract variables into NumPy
    formatting and add additional useful data structures.

    Attrs:
        x_loc: (frames,) array; x location of bird position
        y_loc: (frames,) array; y location of bird position
        theta: (frames,) array; angle of bird position relative to center
        wedges: (frames,) array; which wedge the bird is on
        visit_s: (neurs, visits) array; average spikes each visit
        visit_durs: (visits,) array; frame duration of each visit
        visit_enters: (visits,) array; start frame of each visit
        visit_exits: (visits,) array; end frame of each visit
        visit_wedges: (visits,) array; wedge location of each visit
        cr_sites: (cr,) array: location of each cache/retrieval event
        cr_pokes: (cr,) array: frame of each cache/retrieval poke
        cr_enters: (cr,) array: frame of hop into cache/retrieval event
        cr_exits: (cr,) array: frame of hop out of cache/retrieval event
        was_retrieval: (cr,) array; whether or not cr event was retrieval
        was_cache: (cr,) array; whether or not cr event was cache
        spikes: (neurs, frames) array; number of spikes of each neur per frame


    Args:
        f: h5py File wrapped around the .mat file
    """

    def __init__(self, f):
        if 'XDLC' in f.keys() and 'YDLC' in f.keys():
            self.x_loc = np.array(f['XDLC']['Body']).squeeze()
            self.y_loc = np.array(f['YDLC']['Body']).squeeze()
        else:
            self.x_loc = np.array(f['X']).squeeze()
            self.y_loc = np.array(f['Y']).squeeze()
        self.theta = np.array(f['theta']).squeeze()
        self.wedges = np.array(f['whichWedge']).squeeze()
        self.visit_s = np.array(f['VS']).squeeze().T
        self.visit_durs = np.array(f['VisitDur']).squeeze()
        self.visit_enters = np.array(f['VisitStart']).squeeze().astype(int) - 1
        self.visit_exits = self.visit_enters + self.visit_durs
        self.visit_wedges = np.array(f['VisitWedge']).squeeze()
        self.cr_sites = np.array(f['CacheSites']).squeeze()
        self.cr_pokes = np.array(f['CacheFrames']).squeeze().astype(int) - 1
        self.cr_enters = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
        self.cr_exits = np.array(f['CacheFramesExit']).squeeze().astype(int)
        self.was_retrieval = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
        self.was_cache = np.logical_not(self.was_retrieval)
        self.spikes = np.array(f['S']).T
        self.num_neurs, self.num_frames = self.spikes.shape
        self.fr = get_fr(self.spikes)

    def get_noncache_visits(self): #TODO
        """
        Extracts visits where the bird did not cache or retrieve

        Returns:
            noncache_enters, noncache_exits, noncache_wedges. Same as visits
            counterpart but with c/r visits removed
        """

        visit_enters = self.visit_enters
        visit_exits = self.visit_exits
        visit_wedges = self.visit_wedges
        cache_enters = self.cache_enters
        cache_exits = self.cache_exits
        all_cache_frames = [
            np.arange(enter, exit) for enter, exit in zip(cache_enters, cache_exits)
            ]
        all_cache_frames = np.concatenate(all_cache_frames)
        noncache_idxs = []
        for idx in range(visit_enters.size):
            enter = visit_enters[idx]
            exit = visit_exits[idx]
            if enter in all_cache_frames or exit in all_cache_frames:
                continue
            noncache_idxs.append(idx)
        noncache_enters = visit_enters[noncache_idxs]
        noncache_exits = visit_exits[noncache_idxs]
        noncache_wedges = visit_wedges[noncache_idxs]
        return noncache_enters, noncache_exits, noncache_wedges

    def get_hopcentered_frames(self, window, nanpad=False): #TODO
        cache_enters = self.cache_enters
        cache_exits = self.cache_exits
        noncache_enters = self.noncache_enters
        noncache_exits = self.noncache_exits
        noncache_wedges = self.noncache_wedges
        cache_hops_frames = [
            np.arange(enter-window, enter+window+1) for \
            enter, exit in zip(cache_enters, cache_exits)
            ]
        all_cache_related_frames = [
            np.arange(enter, exit) for enter, exit in zip(cache_enters, cache_exits)
            ]
        all_cache_related_frames.extend(cache_hops_frames)
        all_cache_related_frames = np.concatenate(all_cache_related_frames)
        noncache_hops_frames = []
        noncache_hops_wedges = []
        for idx in range(noncache_enters.size):
            enter = noncache_enters[idx]
            exit = noncache_exits[idx]
            wedge = noncache_wedges[idx]
            start_hop_frame = max(enter - window, 0)
            end_hop_frame = min(enter + window + 1, self.num_frames)
            hops_frames = np.arange(start_hop_frame, end_hop_frame)
            if np.sum(np.isin(hops_frames, all_cache_related_frames)) > 0:
                continue
            if nanpad:
                start_pad = start_hop_frame - (enter-window)
                end_pad = (enter + window + 1) - end_hop_frame
                hops_frames = np.concatenate([
                    np.zeros(start_pad)*np.nan,
                    hops_frames, np.zeros(end_pad)*np.nan
                    ])
                if hops_frames.size != window*2 + 1: import pdb; pdb.set_trace()
            noncache_hops_frames.append(hops_frames)
            noncache_hops_wedges.append(wedge)
        noncache_hops_wedges = np.array(noncache_hops_wedges)
        return cache_hops_frames, noncache_hops_frames, noncache_hops_wedges

    def get_hopcentered_fr(self, window, nanpad=False, fr=None): #TODO
        if fr is None:
            fr = self.fr
        cache_enters = self.cache_enters
        cache_exits = self.cache_exits
        noncache_enters = self.noncache_enters
        noncache_exits = self.noncache_exits
        noncache_wedges = self.noncache_wedges
        cache_hops_frames = [
            np.arange(enter-window, enter+window+1) for \
            enter, exit in zip(cache_enters, cache_exits)
            ]
        cache_hops_fr = [fr[c,:] for c in cache_hops_frames]
        all_cache_related_frames = [
            np.arange(enter, exit) for enter, exit in zip(cache_enters, cache_exits)
            ]
        all_cache_related_frames.extend(cache_hops_frames)
        all_cache_related_frames = np.concatenate(all_cache_related_frames)
        noncache_hops_frames = []
        noncache_hops_wedges = []
        noncache_hops_fr = []
        for idx in range(noncache_enters.size):
            enter = noncache_enters[idx]
            exit = noncache_exits[idx]
            wedge = noncache_wedges[idx]
            start_hop_frame = max(enter - window, 0)
            end_hop_frame = min(enter + window + 1, self.num_frames)
            hops_frames = np.arange(start_hop_frame, end_hop_frame)
            hops_fr = fr[hops_frames, :]
            if np.sum(np.isin(hops_frames, all_cache_related_frames)) > 0:
                continue
            if nanpad:
                continue
                start_pad = start_hop_frame - (enter-window)
                end_pad = (enter + window + 1) - end_hop_frame
                hops_fr = np.concatenate([
                    np.zeros((start_pad, self.num_neurs))*np.nan,
                    fr[hops_frames,:],
                    np.zeros((end_pad, self.num_neurs))*np.nan
                    ])
                if hops_fr.shape[0] != window*2 + 1: import pdb; pdb.set_trace()
            noncache_hops_frames.append(hops_frames)
            noncache_hops_wedges.append(wedge)
            noncache_hops_fr.append(hops_fr)
        noncache_hops_wedges = np.array(noncache_hops_wedges)
        return cache_hops_fr, noncache_hops_fr, noncache_hops_wedges

