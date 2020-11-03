import numpy as np
from dataclasses import dataclass

from analysis.utils import get_fr
from analysis.tuning_measures import fr_index, mutual_info
import analysis.shuffle as shuffle

@dataclass
class EpisodeCalc(object):
    """
    Calculates episode tuning
    """

    window: int
    num_shuffles: int
    threshold: float

    def calc_ep_index(self, exp_data):
        """
        Gets the episode index of each cell. Returns a (cache, neur) array
        """

        cr_hops, noncr_hops = exp_data.get_cr_hops()
        noncr_sites = exp_data.hop_end_wedges[noncr_hops]
        hop_windows = exp_data.get_hop_windows(self.window)
        hop_windows_cr = hop_windows[cr_hops]
        hop_windows_noncr = hop_windows[noncr_hops]
        cr_idx_mat = fr_index.calc_mat(
            exp_data.fr, exp_data.cr_sites, noncr_sites,
            hop_windows_cr, hop_windows_noncr
            )
        cr_idx_mean = np.mean(cr_idx_mat, axis=0)
        significance = np.zeros(exp_data.num_neurs)
        shuff_cr_idx_mat = np.zeros(cr_idx_mat.shape)
        for _ in np.arange(self.num_shuffles):
            shuff_fr = np.zeros(exp_data.fr.shape)
            for neur in np.arange(exp_data.num_neurs):
                shuff_hop_windows_cr, shuff_hop_windows_noncr = \
                    self._shuffle_conditions(hop_windows_cr, hop_windows_noncr)
                shuff_spikes = exp_data.spikes[neur]
                for shuff, idxs in zip(shuff_hop_windows_cr, hop_windows_cr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                for shuff, idxs in zip(shuff_hop_windows_noncr, hop_windows_noncr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                shuff_fr[neur] = get_fr(shuff_spikes)
            shuff_cr_idx_mat = fr_index.calc_mat(
                shuff_fr, exp_data.cr_sites, noncr_sites,
                hop_windows_cr, hop_windows_noncr
                )
            shuff_cr_idx_mean = np.mean(shuff_cr_idx_mat, axis=0)
            significance += (shuff_cr_idx_mean < cr_idx_mean)
        shuff_cr_idx_mat /= self.num_shuffles
        #significance = significance > self.threshold*self.num_shuffles
        significance /= self.num_shuffles
        cr_idx_mat -= shuff_cr_idx_mat
        return cr_idx_mat, significance

    def calc_ep_mi(self, exp_data):
        """ Gets the episode mutual information of each cell."""

        ep_info = np.zeros(exp_data.num_neurs)
        shuffled_ep_info = np.zeros(exp_data.num_neurs)
        significance = np.zeros(exp_data.num_neurs)
        cr_hops, noncr_hops = exp_data.get_cr_hops()
        hop_windows = exp_data.get_hop_windows(self.window)
        hop_windows_cr = hop_windows[cr_hops]
        hop_windows_noncr = hop_windows[noncr_hops]
        conditions = -1*np.ones(exp_data.fr.shape[1])
        for cr in hop_windows_cr:
            conditions[cr[cr != -1]] = 1
        for noncr in hop_windows_noncr:
            conditions[noncr[noncr != -1]] = 0
        ep_info = mutual_info.get_mutual_info(conditions, exp_data.fr)
        for _ in range(self.num_shuffles):
            shuff_fr = np.zeros(exp_data.fr.shape)
            for neur in np.arange(exp_data.num_neurs):
                shuff_hop_windows_cr, shuff_hop_windows_noncr = \
                    self._shuffle_conditions(hop_windows_cr, hop_windows_noncr)
                shuff_spikes = exp_data.spikes[neur]
                for shuff, idxs in zip(shuff_hop_windows_cr, hop_windows_cr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                for shuff, idxs in zip(shuff_hop_windows_noncr, hop_windows_noncr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                shuff_fr[neur] = get_fr(shuff_spikes)
            shuffled_info = mutual_info.get_mutual_info(conditions, shuff_fr)
            shuffled_ep_info += shuffled_info
            significance += (shuffled_info < ep_info)
        shuffled_ep_info /= self.num_shuffles
        ep_info /= shuffled_ep_info
        #significance = (significance > self.threshold*self.num_shuffles).astype(bool)
        significance /= self.num_shuffles
        return ep_info, significance

    def _shuffle_conditions(self, hop_windows_cr, hop_windows_noncr):
        """
        Stacks the cr and non-cr hops on top of each other, shuffles the rows,
        and reassigns each row as a cr or non-cr hop.
        """

        all_hops = np.vstack((hop_windows_cr, hop_windows_noncr))
        np.random.shuffle(all_hops); np.random.shuffle(all_hops)
        shuff_hop_windows_cr = all_hops[:hop_windows_cr.shape[0]]
        shuff_hop_windows_noncr = all_hops[:hop_windows_noncr.shape[0]]
        return shuff_hop_windows_cr, shuff_hop_windows_noncr

