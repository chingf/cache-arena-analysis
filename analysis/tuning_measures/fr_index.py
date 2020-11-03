import numpy as np

def calc_mat(fr, cr_sites, noncr_sites, hop_windows_cr, hop_windows_noncr):
    """
    Forms the f.r. index matrix out of rows.

    Args:
        fr: (neurs, frames) array
        cr_sites: (cr,) array of locations of cache/retrievals
        noncr_sites: (noncr,) array of locations of non-cache/retrieval hops
        hop_windows_cr: (cr, window) array
        hop_windows_noncr: (non-cr, window) array
    Returns:
        (cr, neurs) matrix of firing rate indices.
    """

    num_neurs, _ = fr.shape
    cr_idx_mat = np.zeros((cr_sites.size, num_neurs))
    for cr, cr_wedge in enumerate(cr_sites):
        in_condition = hop_windows_cr[cr]
        in_condition = in_condition[in_condition != -1]
        out_conditions = hop_windows_noncr[noncr_sites == cr_wedge]
        out_conditions = out_conditions[~(out_conditions == -1).any(axis=1)]
        cr_idx_row = calc_row(fr, in_condition, out_conditions)
        cr_idx_mat[cr,:] = cr_idx_row
    return cr_idx_mat

def calc_row(fr, in_condition, out_conditions):
    """
    Compares the firing at an in-condition (episode) versus out-of-condition
    moments.

    Args:
        fr: (neurs, frames) array
        in_condition: (window,) array containing the frames of the episode
        out_conditions: (out-of-condition hops, window) array containing
            the frames of out-of-condition hops
    Returns:
        (neurs,) array of firing rate indices. A row of the f.r. index matrix
    """

    in_condition_frs = np.mean(fr[:,in_condition], axis=1) # (neurs,)
    out_condition_frs = []
    for out_condition in out_conditions:
        out_condition_fr = np.mean(fr[:, out_condition], axis=1) # (neurs,)
        out_condition_frs.append(out_condition_fr)
    out_condition_frs = np.array(out_condition_frs) # (cond, neurs)
    fr_idx = np.sum(out_condition_frs < in_condition_frs, axis=0)
    fr_idx = fr_idx/out_condition_frs.shape[0]
    return fr_idx

