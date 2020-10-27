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
        self.spikes = np.array(f['S']).T
        self.num_neurs, self.num_frames = self.spikes.shape
        self.fr = get_fr(self.spikes)
        self.visit_s = np.array(f['VS']).squeeze().T
        self.visit_durs = np.array(f['VisitDur']).squeeze().astype(int)
        self.visit_enters = np.array(f['VisitStart']).squeeze().astype(int) - 1
        self.visit_exits = self.visit_enters + self.visit_durs + 1
        self.visit_wedges = np.array(f['VisitWedge']).squeeze().astype(int)

        if "LMN" in f.filename:
            self.event_sites = np.array(f['EventSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['EventFrames']).squeeze().astype(int) - 1
            self.event_enters = np.array(f['EventFramesEnter']).squeeze().astype(int) - 1
            self.event_exits = np.array(f['EventFramesExit']).squeeze().astype(int) + 1
            event_labels = np.array(f['TypeOfEvent']).squeeze()
            self.cache_event = (event_labels == 1)
            self.retriev_event = (event_labels == 2)
            self.check_event = (event_labels == 3)
            self._align_checks()
        else:
            self.event_sites = np.array(f['CacheSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['CacheFrames']).squeeze().astype(int) - 1
            self.event_enters = np.array(f['CacheFramesEnter']).squeeze().astype(int) - 1
            self.event_exits = np.array(f['CacheFramesExit']).squeeze().astype(int)
            self.retriev_event = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
            self.cache_event = np.logical_not(self.retriev_event)
            self.check_event = np.zeros(self.event_sites.size).astype(bool)
        self._remove_repeated_events()
        self._sort_events()
        self._label_cache_present()

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

    def get_crch_visits(self):
        """
        Labels each visit as cache/retrieval/check, or non-c/r/ch.

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
        _, ch_visits, _ = np.intersect1d(
            self.visit_enters, self.event_enters[self.check_event],
            return_indices = True
            )
        crch_visits = np.concatenate((c_visits, r_visits, ch_visits))
        noncr_visits = np.arange(self.visit_enters.size)
        noncr_visits = np.setdiff1d(noncr_visits, crch_visits)
        return c_visits, r_visits, ch_visits, noncr_visits

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

    def _align_checks(self):
        """
        Temporary method to fix incorrectly formatted data in LMN73. Throws out
        the current (incorrect) estimate of check frame enters and ensures they
        are aligned to a visit frame enter.
        """

        visit_idxs = np.digitize(self.event_enters, self.visit_enters) - 1
        visit_wedges = self.visit_wedges[visit_idxs]
        valid_indices = (self.event_sites == visit_wedges)
        self.event_sites = self.event_sites[valid_indices]
        self.event_pokes = self.event_pokes[valid_indices]
        self.event_enters = self.event_enters[valid_indices]
        self.event_exits = self.event_exits[valid_indices]
        self.cache_event = self.cache_event[valid_indices]
        self.retriev_event = self.retriev_event[valid_indices]
        self.check_event = self.check_event[valid_indices]
        incorrect_exits = np.logical_not(np.isin(self.event_exits, self.visit_exits))
        if np.sum(incorrect_exits) > 0:
            correct_idxs = np.digitize(
                self.event_exits[incorrect_exits], self.visit_exits
                ) - 1
            self.event_exits[incorrect_exits] = self.visit_exits[correct_idxs]

    def _label_cache_present(self):
        """
        Creates a (visit,) array that indicates, for each visit, whether a cache
        was present at the site of the visit
        """

        cache_present = np.zeros((self.visit_wedges.size, 16)).astype(bool)
        for c_enter, c_site in zip(
            self.event_enters[self.cache_event],
            self.event_sites[self.cache_event]
            ):
            for r_enter, r_site in zip(
                self.event_enters[self.retriev_event],
                self.event_sites[self.retriev_event]
                ):
                    if c_site != r_site: continue
                    if r_enter < c_enter: continue
                    visits_before_retriev = self.visit_enters < r_enter
                    visits_after_cache = self.visit_enters > c_enter
                    visits_at_site = self.visit_wedges == c_site
                    in_between_visits = np.logical_and(
                        visits_after_cache, visits_before_retriev
                        )
                    cache_present[in_between_visits, c_site-1] = True
                    break
        self.cache_present = cache_present
