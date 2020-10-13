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
        visit_was_retrieval: (visits,) array; whether visit was retrieval
        visit_was_cache: (visits,) array; whether visit was cache
        cr_sites: (cr,) array: location of each cache/retrieval event
        cr_pokes: (cr,) array: frame of each cache/retrieval poke
        cr_enters: (cr,) array: frame of hop into cache/retrieval event
        cr_exits: (cr,) array: frame of hop out of cache/retrieval event
        cr_was_retrieval: (cr,) array; whether or not cr event was retrieval
        cr_was_cache: (cr,) array; whether or not cr event was cache
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
        self.visit_wedges = np.array(f['VisitWedge']).squeeze().astype(int)
        self.cr_sites = np.array(f['CacheSites']).squeeze().astype(int)
        self.cr_pokes = np.array(f['CacheFrames']).squeeze().astype(int) - 1
        self.cr_enters = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
        self.cr_exits = np.array(f['CacheFramesExit']).squeeze().astype(int)
        self.cr_was_retrieval = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
        self.cr_was_cache = np.logical_not(self.cr_was_retrieval)
        self._remove_repeated_crs()
        self._sort_crs()
        self.spikes = np.array(f['S']).T
        self.num_neurs, self.num_frames = self.spikes.shape
        self.fr = get_fr(self.spikes)

    def get_cr_visits(self):
        """
        Labels each visit as cache/retrieval, or non-cr.

        Returns:
            cr_visits, noncr_visits: arrays that index into visits to extract
            each type of visit. The size of both arrays stacked together
            should be (visits,)
        """

        _, cr_visits, _ = np.intersect1d(
            self.visit_enters, self.cr_enters, return_indices = True
            )
        noncr_visits = np.arange(self.visit_enters.size)
        noncr_visits = np.setdiff1d(noncr_visits, cr_visits)
        return cr_visits, noncr_visits

    def get_hopcentered_visits(self, window):
        """
        Collects visits centered on hops into the site for a given window size.

        Returns:
            (visits, window) array with NaN-padding where indices extend out of
            range.
        """

        hopcentered_visits = np.array([
            np.arange(enter-window, enter+window+1) for \
            enter, exit in zip(self.visit_enters, self.visit_exits)
            ])
        hopcentered_visits[hopcentered_visits < 0] = -1
        hopcentered_visits[hopcentered_visits >= self.num_frames] = -1
        return hopcentered_visits

    def _remove_repeated_crs(self):
        unique_enters = []
        keep_cr = np.ones(self.cr_enters.size).astype(bool)
        for idx, enter in enumerate(self.cr_enters):
            if enter in unique_enters:
                keep_cr[idx] = False
            else:
                unique_enters.append(enter)
        self.cr_sites = self.cr_sites[keep_cr]
        self.cr_pokes = self.cr_pokes[keep_cr]
        self.cr_enters = self.cr_enters[keep_cr]
        self.cr_exits = self.cr_exits[keep_cr]
        self.cr_was_retrieval = self.cr_was_retrieval[keep_cr]
        self.cr_was_cache = self.cr_was_cache[keep_cr]

    def _sort_crs(self):
        sorting = np.argsort(self.cr_enters)
        self.cr_sites = self.cr_sites[sorting]
        self.cr_pokes = self.cr_pokes[sorting]
        self.cr_enters = self.cr_enters[sorting]
        self.cr_exits = self.cr_exits[sorting]
        self.cr_was_retrieval = self.cr_was_retrieval[sorting]
        self.cr_was_cache = self.cr_was_cache[sorting]

