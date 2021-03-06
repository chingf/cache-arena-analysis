import pickle
import numpy as np
from math import pi
from scipy.signal import find_peaks
from analysis.config import arena_params, pickle_dir
from analysis.utils import cart2pol, get_fr, get_max_consecutive, in_ellipse
from analysis.utils import get_consecutive

class ExpData(object):
    """
    Given data in Emily's file format, will extract variables into NumPy
    formatting and add additional useful data structures.

    Attrs:
        x: (frames,) array; x location of bird position
        y: (frames,) array; y location of bird position
        x_head: (frames,) array; x location of bird head
        y_head: (frames,) array; y location of bird head
        thetas: (frames,) array; angle of bird position relative to center
        wedges: (frames,) array; which wedge the bird is on
        speeds: (frames,) array; speed of bird in pixels/frame
        hops: (hops,) array; frame of peak speed in each hop
        hop_starts: (hops,) array; frame of trough speed before hop
        hop_ends: (hops,) array; frame of trough speed after hop
        hop_start_wedges: (hops,) array; which wedge each hop came from
        hop_end_wedges: (hops,) array; which wedge each hop ended on
        event_sites: (events,) array: location of each cache/retrieval event
        event_pokes: (events,) array: frame of each cache/retrieval poke
        event_hops: (events,) array: frame of hop into cache/retrieval event
        cache_event: (events,) array; whether event was cache
        retriev_event: (events,) array; whether event was retrieval
        check_event: (events,) array; whether event was check
        spikes: (neurs, frames) array; number of spikes of each neur per frame

    Args:
        f: h5py File wrapped around the .mat file
    """

    def __init__(self, f, min_hop_gap=0):
        if 'XDLC' in f.keys() and 'YDLC' in f.keys():
            self.x = np.array(f['XDLC']['Body']).squeeze()
            self.y = np.array(f['YDLC']['Body']).squeeze()
            self.x_head = np.array(f['XDLC']['Head']).squeeze()
            self.y_head = np.array(f['YDLC']['Head']).squeeze()
        else:
            self.x = np.array(f['X']).squeeze()
            self.y = np.array(f['Y']).squeeze()
            self.x_head = None 
            self.y_head = None
        self.spikes = np.array(f['S']).T
        self.num_neurs, self.num_frames = self.spikes.shape
        self.fr = get_fr(self.spikes)
        if "LMN" in f.filename:
            self.event_sites = 17 - np.array(f['EventSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['EventFrames']).squeeze().astype(int) - 1
            event_labels = np.array(f['TypeOfEvent']).squeeze()
            self.cache_event = (event_labels == 1)
            self.retriev_event = (event_labels == 2)
            self.check_event = (event_labels == 3)
        else:
            self.event_sites = 17 - np.array(f['CacheSites']).squeeze().astype(int)
            self.event_pokes = np.array(f['CacheFrames']).squeeze().astype(int) - 1
            self.retriev_event = np.array(f['ThisWasRetrieval']).squeeze().astype(bool)
            self.cache_event = np.logical_not(self.retriev_event)
            self.check_event = np.zeros(self.event_sites.size).astype(bool)
        self._set_positional_variables(f.filename)
        self._set_hop_visits()
        self._remove_super_short_hops()
        if min_hop_gap != 0:
            self._merge_hops(min_hop_gap)
        self._set_event_labels()
        self._remove_repeated_events()
        self._sort_events()
        self._label_cache_present()

    def get_cr_hops(self):
        """
        Labels each hop as cache/retrieval, or non-cache/retrieval.

        Returns:
            cr_hops, noncr_hops: arrays that index into hops to extract
            each type of hop. The size of both arrays stacked together
            should be (hops,)
        """

        _, c_hops, _ = np.intersect1d(
            self.hops, self.event_hops[self.cache_event],
            return_indices = True
            )
        _, r_hops, _ = np.intersect1d(
            self.hops, self.event_hops[self.retriev_event],
            return_indices = True
            )
        cr_hops = np.concatenate((c_hops, r_hops))
        noncr_hops = np.arange(self.hops.size)
        noncr_hops = np.setdiff1d(noncr_hops, cr_hops)
        return c_hops, r_hops, noncr_hops

    def get_crch_hops(self):
        """
        Labels each hop as cache/retrieval/check, or non-c/r/ch.

        Returns:
            cr_hops, noncr_hops: arrays that index into hops to extract
            each type of hop. The size of both arrays stacked together
            should be (hops,)
        """

        _, c_hops, _ = np.intersect1d(
            self.hops, self.event_hops[self.cache_event],
            return_indices = True
            )
        _, r_hops, _ = np.intersect1d(
            self.hops, self.event_hops[self.retriev_event],
            return_indices = True
            )
        _, ch_hops, _ = np.intersect1d(
            self.hops, self.event_hops[self.check_event],
            return_indices = True
            )
        crch_hops = np.concatenate((c_hops, r_hops, ch_hops))
        noncrch_hops = np.arange(self.hops.size)
        noncrch_hops = np.setdiff1d(noncrch_hops, crch_hops)
        return c_hops, r_hops, ch_hops, noncrch_hops

    def get_hop_windows(self, window, center_frames=None):
        """
        Collects windows centered on hops into the site
        
        Args:
            window: number of frames before and after the center frame to collect
            center_frames: (hops,) array of frames to align to

        Returns:
            (hops, window) array with (-1)-padding where indices extend out of
            range.
        """

        center_frames = self.hops if center_frames is None else center_frames
        hop_windows = np.array([
            np.arange(c-window, c+window+1) for c in center_frames
            ])
        hop_windows[hop_windows < 0] = -1
        hop_windows[hop_windows >= self.num_frames] = -1
        return hop_windows

    def get_putative_checks(self, window=20*2):
        """ Labels each hop as a putative check or not. """

        with open(pickle_dir / "check_params.p", "rb") as f:
            params = pickle.load(f)
        body_centers = params["body_centers"]
        body_semiaxes = params["body_semiaxes"]
        head_centers = params["head_centers"]
        head_semiaxes = params["head_semiaxes"]
        thresholds = params["thresholds"]
        putative_checks = np.zeros(self.hops.size)
        for hop_idx, hop in enumerate(self.hops):
            hop_site = self.hop_end_wedges[hop_idx]
            hop_start = hop - window
            hop_end = hop + window
            if hop_site == 17:
                continue
            if hop_start < 0 or hop_end >= self.num_frames:
                putative_checks[hop_idx] = True
                continue
            head_in_roi = in_ellipse(
                self.x_head[hop_start:hop_end],
                self.y_head[hop_start:hop_end],
                head_centers[hop_site - 1], head_semiaxes[hop_site - 1]
                )
            body_in_roi = in_ellipse(
                self.x[hop_start:hop_end],
                self.y[hop_start:hop_end],
                body_centers[hop_site - 1], body_semiaxes[hop_site - 1]
                )
            both_in_roi = np.logical_and(head_in_roi, body_in_roi)
            if get_max_consecutive(both_in_roi) > thresholds[hop_site - 1]:
                putative_checks[hop_idx] = True
        return putative_checks

    def _set_positional_variables(self, filename):
        """ Calculates position theta, wedges, speed. """

        bird_key = "RBY" if "RBY" in filename else "LMN"
        arena_center = arena_params[bird_key]["arena_center"]
        w17_radius = arena_params[bird_key]["w17_radius"]
        rhos, thetas = cart2pol(self.x - arena_center[0], self.y - arena_center[1])
        thetas[thetas < 0] = 2*pi + thetas[thetas < 0] # Format to [0, 2pi]
        wedges = np.digitize(thetas, np.arange(0, 2*pi, 2*pi/16))
        wedges[rhos < w17_radius] = 17
        delta_x = self.x[1:] - self.x[:-1]
        delta_y = self.y[1:] - self.y[:-1]
        speeds = np.sqrt(np.square(delta_x) + np.square(delta_y))
        speeds = np.insert(speeds, 0, 0)
        self.thetas = thetas
        self.wedges = wedges
        self.speeds = speeds

    def _set_hop_visits(self):
        """
        Finds hops by searching for velocity peaks between wedge changes.
        """

        pk_thresh = 10
        pk_bottom = 3
        hops = []; hop_starts = []; hop_ends = []
        hop_start_wedges = []; hop_end_wedges = []
        pks, _ = find_peaks(self.speeds, height=pk_thresh)
        troughs = self.speeds <= pk_bottom
        troughs_idxs = np.argwhere(troughs).squeeze()
        for pk in pks:
            pk_start = troughs_idxs[troughs_idxs < pk][-1]
            pk_end = troughs_idxs[troughs_idxs > pk][0]
            if self.wedges[pk_start] != self.wedges[pk_end]:
                hops.append(pk)
                hop_starts.append(pk_start)
                hop_ends.append(pk_end)
                hop_start_wedges.append(self.wedges[pk_start])
                hop_end_wedges.append(self.wedges[pk_end])
        _, unique_idxs = np.unique(hop_starts, return_index=True)
        _, test_idxs = np.unique(hop_ends, return_index=True)
        assert(sum(unique_idxs == test_idxs) == unique_idxs.size)
        self.hops = np.array(hops)[unique_idxs]
        self.hop_starts = np.array(hop_starts)[unique_idxs]
        self.hop_ends = np.array(hop_ends)[unique_idxs]
        self.hop_start_wedges = np.array(hop_start_wedges)[unique_idxs]
        self.hop_end_wedges = np.array(hop_end_wedges)[unique_idxs]

    def _remove_super_short_hops(self):
        """
        Removes hops that only last a couple of frames. These are likely
        the result of DLC labeling errors or dropped video frames.
        """

        valid_hops = np.logical_not((self.hop_ends - self.hop_starts) < 3)
        self.hops = self.hops[valid_hops]
        self.hop_starts = self.hop_starts[valid_hops]
        self.hop_ends = self.hop_ends[valid_hops]
        self.hop_start_wedges = self.hop_start_wedges[valid_hops]
        self.hop_end_wedges = self.hop_end_wedges[valid_hops]

    def _merge_hops(self, min_gap):
        """
        Merge hops if the bird pauses for less than some threshold between
        hops.
        """

        hop_diffs = self.hop_starts[1:] - self.hop_ends[:-1]
        hop_diffs = np.concatenate([hop_diffs, [0]])
        merge_starts, merge_ends = get_consecutive(hop_diffs < min_gap)
        new_hops = []; new_hop_starts = []; new_hop_ends = [];
        new_hop_start_wedges = []; new_hop_end_wedges = [];
        for merge_start, merge_end in zip(merge_starts, merge_ends):
            merge_mid = (merge_start + merge_end)//2
            new_hops.append(self.hops[merge_mid])
            new_hop_starts.append(self.hop_starts[merge_start])
            new_hop_ends.append(self.hop_ends[merge_end])
            new_hop_start_wedges.append(self.hop_start_wedges[merge_start])
            new_hop_end_wedges.append(self.hop_end_wedges[merge_end])
        self.hops = np.array(new_hops)
        self.hop_starts = np.array(new_hop_starts)
        self.hop_ends = np.array(new_hop_ends)
        self.hop_start_wedges = np.array(new_hop_start_wedges)
        self.hop_end_wedges = np.array(new_hop_end_wedges)

    def _set_event_labels(self):
        """
        Finds the corresponding hop for each cache/retrieval/check event.
        """

        hop_idxs = np.digitize(self.event_pokes, self.hops) - 1
        hop_wedges = self.hop_end_wedges[hop_idxs]
        valid_indices = (self.event_sites == hop_wedges)
        diffs = self.event_pokes - self.hops[hop_idxs]
        self.event_sites = self.event_sites[valid_indices]
        self.event_pokes = self.event_pokes[valid_indices]
        self.event_hops = self.hops[hop_idxs][valid_indices]
        self.cache_event = self.cache_event[valid_indices]
        self.retriev_event = self.retriev_event[valid_indices]
        self.check_event = self.check_event[valid_indices]

    def _remove_repeated_events(self):
        """ Removes duplicate events from incorrect labeling. """

        unique_hops = []
        keep_event = np.ones(self.event_hops.size).astype(bool)
        for idx, hop in enumerate(self.event_hops):
            if hop in unique_hops:
                keep_event[idx] = False
            else:
                unique_hops.append(hop)
        self.event_sites = self.event_sites[keep_event]
        self.event_pokes = self.event_pokes[keep_event]
        self.event_hops = self.event_hops[keep_event]
        self.cache_event = self.cache_event[keep_event]
        self.retriev_event = self.retriev_event[keep_event]
        self.check_event = self.check_event[keep_event]

    def _sort_events(self):
        """ Sorts the event arrays so they are in chronological order. """

        sorting = np.argsort(self.event_hops)
        self.event_sites = self.event_sites[sorting]
        self.event_pokes = self.event_pokes[sorting]
        self.event_hops = self.event_hops[sorting]
        self.cache_event = self.cache_event[sorting]
        self.retriev_event = self.retriev_event[sorting]
        self.check_event = self.check_event[sorting]

    def _label_cache_present(self):
        """
        Creates a (hops,) array that indicates, for each hop, whether a cache
        was present at the site of the hop
        """

        cache_present = np.zeros((self.hops.size, 16)).astype(bool)
        for c_hop, c_site in zip(
            self.event_hops[self.cache_event],
            self.event_sites[self.cache_event]
            ):
            for r_hop, r_site in zip(
                self.event_hops[self.retriev_event],
                self.event_sites[self.retriev_event]
                ):
                    if c_site != r_site: continue
                    if r_hop < c_hop: continue
                    hops_before_retriev = self.hops < r_hop
                    hops_after_cache = self.hops > c_hop
                    hops_at_site = self.hop_end_wedges == c_site
                    in_between_hops = np.logical_and(
                        hops_after_cache, hops_before_retriev
                        )
                    cache_present[in_between_hops, c_site-1] = True
                    break
        self.cache_present = cache_present

