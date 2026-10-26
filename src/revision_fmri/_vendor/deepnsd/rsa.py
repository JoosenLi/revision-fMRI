# Exact function excerpts from DeepNSD; see NOTICE.md and manifest.json.
import numpy as np
from warnings import warn
from scipy.stats import pearsonr, spearmanr


def compare_rdms(rdm1, rdm2, dist_type='pearson'):
    rdm1_triu = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    rdm2_triu = rdm2[np.triu_indices(rdm2.shape[0], k=1)]
    
    if (np.sum(np.isnan(rdm1_triu)) > 0 or 
        np.sum(np.isnan(rdm2_triu)) > 0):
        warn('Warning: RDMs contain NaNs...',
             'Returning None.'); return None
    
    if dist_type == 'pearson':
        return pearsonr(rdm1_triu, rdm2_triu)[0]
    if dist_type == 'spearman':
        return spearmanr(rdm1_triu, rdm2_triu)[0]
