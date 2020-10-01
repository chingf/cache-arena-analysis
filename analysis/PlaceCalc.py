import numpy as np
from dataclasses import dataclass

import analysis.tuning_measures.mutual_info as mi
import analysis.shuffle as shuffle
from analysis.utils import get_fr, gen_2d_bins

@dataclass
class PlaceCalc(object):
    """
    Calculates spatial tuning in either 1D (wedges) or 2D
    """

    num_shuffles: int
    threshold: float
    dimension: int
    bins_2d: int

    def calc_place_cells(self, exp_data):
        """ Gets the spatial mutual information of each cell."""

        spatial_info = np.zeros(exp_data.num_neurs)
        shuffled_spatial_info = np.zeros(exp_data.num_neurs)
        significance = np.zeros(exp_data.num_neurs)
        conditions, fr, bool_mask = self.format_data(exp_data)
        spatial_info = mi.get_mutual_info(conditions, fr)
        for _ in range(self.num_shuffles):
            shuffled_fr = get_fr(shuffle.circular(exp_data.spikes))
            shuffled_info = mi.get_mutual_info(conditions, shuffled_fr[:,bool_mask])
            shuffled_spatial_info += shuffled_info
            significance += (shuffled_info < spatial_info)
        shuffled_spatial_info /= self.num_shuffles
        spatial_info /= shuffled_spatial_info
        significance = (significance > self.threshold*self.num_shuffles).astype(bool)
        return spatial_info, significance

    def format_data(self, exp_data):
        """ Extracts the 1D or 2D binning for the data. """

        if self.dimension == 1:
            conditions = exp_data.wedges[exp_data.wedges != 17]
            fr = exp_data.fr[:,exp_data.wedges != 17]
            bool_mask = (exp_data.wedges != 17).astype(bool)
        elif self.dimension == 2:
            _, conditions = gen_2d_bins(
                exp_data.x_loc, exp_data.y_loc, self.bins_2d
                )
            fr = exp_data.fr
            bool_mask = np.ones(fr.shape[1]).astype(bool)
        else:
            raise ValueError("Only 1D or 2D spatial information allowed.")
        return conditions, fr, bool_mask
