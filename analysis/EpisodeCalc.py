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
        """ Gets the episode index of each cell. """

        cr_visits, noncr_visits = exp_data.get_cr_visits()
        noncr_sites = exp_data.visit_wedges[noncr_visits]
        hopcentered_visits = exp_data.get_hopcentered_visits(self.window)
        hopcentered_cr = hopcentered_visits[cr_visits]
        hopcentered_noncr = hopcentered_visits[noncr_visits]
        cr_idx_mat = fr_index.calc_mat(
            exp_data.fr, exp_data.cr_sites, noncr_sites,
            hopcentered_cr, hopcentered_noncr
            )
        cr_idx_mean = np.mean(cr_idx_mat, axis=0)
        significance = np.zeros(exp_data.num_neurs)
        shuff_cr_idx_mat = np.zeros(cr_idx_mat.shape)
        for _ in np.arange(self.num_shuffles):
            shuff_fr = np.zeros(exp_data.fr.shape)
            for neur in np.arange(exp_data.num_neurs):
                shuff_hopcentered_cr, shuff_hopcentered_noncr = \
                    self._shuffle_conditions(hopcentered_cr, hopcentered_noncr)
                shuff_spikes = exp_data.spikes[neur]
                for shuff, idxs in zip(shuff_hopcentered_cr, hopcentered_cr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                for shuff, idxs in zip(shuff_hopcentered_noncr, hopcentered_noncr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                shuff_fr[neur] = get_fr(shuff_spikes)
            shuff_cr_idx_mat = fr_index.calc_mat(
                shuff_fr, exp_data.cr_sites, noncr_sites,
                hopcentered_cr, hopcentered_noncr
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
        cr_visits, noncr_visits = exp_data.get_cr_visits()
        hopcentered_visits = exp_data.get_hopcentered_visits(self.window)
        hopcentered_cr = hopcentered_visits[cr_visits]
        hopcentered_noncr = hopcentered_visits[noncr_visits]
        conditions = -1*np.ones(exp_data.fr.shape[1])
        for cr in hopcentered_cr:
            conditions[cr[cr != -1]] = 1
        for noncr in hopcentered_noncr:
            conditions[noncr[noncr != -1]] = 0
        ep_info = mutual_info.get_mutual_info(conditions, exp_data.fr)
        for _ in range(self.num_shuffles):
            shuff_fr = np.zeros(exp_data.fr.shape)
            for neur in np.arange(exp_data.num_neurs):
                shuff_hopcentered_cr, shuff_hopcentered_noncr = \
                    self._shuffle_conditions(hopcentered_cr, hopcentered_noncr)
                shuff_spikes = exp_data.spikes[neur]
                for shuff, idxs in zip(shuff_hopcentered_cr, hopcentered_cr):
                    shuff_spikes[idxs] = exp_data.spikes[neur, shuff]
                for shuff, idxs in zip(shuff_hopcentered_noncr, hopcentered_noncr):
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

    def _shuffle_conditions(self, hopcentered_cr, hopcentered_noncr):
        """
        Stacks the cr and non-cr visits on top of each other, shuffles the rows,
        and reassigns each row as a cr or non-cr visit.
        """

        all_visits = np.vstack((hopcentered_cr, hopcentered_noncr))
        np.random.shuffle(all_visits); np.random.shuffle(all_visits)
        shuff_hopcentered_cr = all_visits[:hopcentered_cr.shape[0]]
        shuff_hopcentered_noncr = all_visits[:hopcentered_noncr.shape[0]]
        return shuff_hopcentered_cr, shuff_hopcentered_noncr

