import numpy as np
from analysis.utils import *

class ExpData(object):
    """
    Given data in Emily's file format, will extract variables into NumPy
    formatting and add additional useful data structures.

    Attrs:
        x: (frames,) array; x location of bird position
        y: (frames,) array; y location of bird position
        thetas: (frames,) array; angle of bird position relative to center
        wedges: (frames,) array; which wedge the bird is on
        visit_s: (neurs, visits) array; average spikes each visit
        visit_durs: (visits,) array; frame duration of each visit
        visit_enters: (visits,) array; start frame of each visit
        visit_exits: (visits,) array; end frame of each visit
        visit_wedges: (visits,) array; wedge location of each visit
        visit_was_retrieval: (visits,) array; whether visit was retrieval
        visit_was_cache: (visits,) array; whether visit was cache
        event_sites: (events,) array: location of each cache/retrieval event
        event_pokes: (events,) array: frame of each cache/retrieval poke
        event_enters: (events,) array: frame of hop into cache/retrieval event
        event_exits: (events,) array: frame of hop out of cache/retrieval event
        cache_event: (events,) array; whether event was cache
        retriev_event: (events,) array; whether event was retrieval
        check_event: (events,) array; whether event was check
        spikes: (neurs, frames) array; number of spikes of each neur per frame


    Args:
        f: h5py File wrapped around the .mat file
    """

    def __init__(self, f):
        if 'XDLC' in f.keys() and 'YDLC' in f.keys():
            self.x = np.array(f['XDLC']['Body']).squeeze()
            self.y = np.array(f['YDLC']['Body']).squeeze()
        else:
            self.x = np.array(f['X']).squeeze()
            self.y = np.array(f['Y']).squeeze()
        self.thetas = np.array(f['theta']).squeeze()
        self.wedges = np.array(f['whichWedge']).squeeze()
        self.visit_s = np.array(f['VS']).squeeze().T
        self.visit_durs = np.array(f['VisitDur']).squeeze().astype(int)
        self.visit_enters = np.array(f['VisitStart']).squeeze().astype(int) - 1
        self.visit_exits = self.visit_enters + self.visit_durs
        self.visit_wedges = np.array(f['VisitWedge']).squeeze().astype(int)

        if "LMN" in f.filename:
            self.event_sites = np.array(f['EventSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['EventFrames']).squeeze().astype(int) - 1
            self.event_enters = np.array(f['EventFramesEnter']).squeeze().astype(int) - 1
            self.event_exits = np.array(f['EventFramesExit']).squeeze().astype(int)
            event_labels = np.array(f['TypeOfEvent']).squeeze()
            self.cache_event = (event_labels == 1)
            self.retriev_event = (event_labels == 2)
            self.check_event = (event_labels == 3)
        else:
            self.event_sites = np.array(f['CacheSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['CacheFrames']).squeeze().astype(int) - 1
            self.event_enters = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
            self.event_exits = np.array(f['CacheFramesExit']).squeeze().astype(int)
            self.retriev_event = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
            self.cache_event = np.logical_not(self.retriev_event)
            self.check_event = np.zeros(event_sites.size).astype(bool)

        self._remove_repeated_events()
        self._sort_events()
        self.spikes = np.array(f['S']).T
        self.num_neurs, self.num_frames = self.spikes.shape
        self.fr = get_fr(self.spikes)

    def get_cr_visits(self):
        """
        Labels each visit as cache/retrieval, or non-cache/retrieval.

        Returns:
            cr_visits, noncr_visits: arrays that index into visits to extract
            each type of visit. The size of both arrays stacked together
            should be (visits,)
        """

        _, c_visits, _ = np.intersect1d(
            self.visit_enters, self.event_enters[self.cache_event],
            return_indices = True
            )
        _, r_visits, _ = np.intersect1d(
            self.visit_enters, self.event_enters[self.retriev_event],
            return_indices = True
            )
        cr_visits = np.concatenate((c_visits, r_visits))
        noncr_visits = np.arange(self.visit_enters.size)
        noncr_visits = np.setdiff1d(noncr_visits, cr_visits)
        return c_visits, r_visits, noncr_visits

    def get_hopcentered_visits(self, window):
        """
        Collects visits centered on hops into the site for a given window size.

        Returns:
            (visits, window) array with (-1)-padding where indices extend out of
            range.
        """

        hopcentered_visits = np.array([
            np.arange(enter-window, enter+window+1) for \
            enter, exit in zip(self.visit_enters, self.visit_exits)
            ])
        hopcentered_visits[hopcentered_visits < 0] = -1
        hopcentered_visits[hopcentered_visits >= self.num_frames] = -1
        return hopcentered_visits

    def _remove_repeated_events(self):
        unique_enters = []
        keep_event = np.ones(self.event_enters.size).astype(bool)
        for idx, enter in enumerate(self.event_enters):
            if enter in unique_enters:
                keep_event[idx] = False
            else:
                unique_enters.append(enter)
        self.event_sites = self.event_sites[keep_event]
        self.event_pokes = self.event_pokes[keep_event]
        self.event_enters = self.event_enters[keep_event]
        self.event_exits = self.event_exits[keep_event]
        self.cache_event = self.cache_event[keep_event]
        self.retriev_event = self.retriev_event[keep_event]
        self.check_event = self.check_event[keep_event]

    def _sort_events(self):
        sorting = np.argsort(self.event_enters)
        self.event_sites = self.event_sites[sorting]
        self.event_pokes = self.event_pokes[sorting]
        self.event_enters = self.event_enters[sorting]
        self.event_exits = self.event_exits[sorting]
        self.cache_event = self.cache_event[sorting]
        self.retriev_event = self.retriev_event[sorting]
        self.check_event = self.check_event[sorting]

