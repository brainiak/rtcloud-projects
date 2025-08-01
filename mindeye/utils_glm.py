import pandas as pd
from pathlib import Path
import numpy as np
import nilearn
from typing import List, Optional, Union, Tuple

def load_glmsingle_hrf_library(tsv_path, library_time_length=32.0):
    """
    Parameters
    ----------
    tsv_path : str or Path
        Path to the TSV file with shape (n_timepoints, 20).
    library_time_length : float
        Total duration (seconds) covered by the library. GLMsingle default is 32s.

    Returns
    -------
    base_time : (n_timepoints,) ndarray
        Time axis for the library, from 0 to library_time_length inclusive.
    hrfs0 : (n_timepoints, 20) ndarray
        HRF library; each column is one HRF, typically peak-normalized to 1.
    """
    tsv_path = Path(tsv_path)
    hrfs0 = np.loadtxt(tsv_path, delimiter="\t")  # shape (T,20)
    if hrfs0.ndim != 2 or hrfs0.shape[1] != 20:
        raise ValueError(f"Expected (n_timepoints, 20) in {tsv_path}, got {hrfs0.shape}")
    n_timepoints = hrfs0.shape[0]
    # Build time axis from 0..32s (inclusive) with n_timepoints samples
    base_time = np.linspace(0.0, library_time_length, n_timepoints, endpoint=True)
    # Safety: ensure each column is peak-normalized (GLMsingle convention)
    max_per_col = np.maximum(np.abs(hrfs0).max(axis=0), 1e-12)
    hrfs0 = hrfs0 / max_per_col
    return base_time, hrfs0



def hrf_i_factory(i, base_time, hrfs, library_time_length=32.0):
    """
    Create a nilearn-compatible HRF function for the i-th GLMsingle HRF (1..20).

    Usage:
        hrf_fn = hrf_i_factory(7)
        model = FirstLevelModel(t_r=TR, hrf_model=hrf_fn, ...)

    The returned function will be called by nilearn as:
        hrf_fn(t_r, oversampling=..., time_length=...)
    and must return a 1-D array sampled on the grid defined by t_r/oversampling.
    """
    if not (1 <= i <= 20):
        raise ValueError("HRF index i must be in 1..20 (GLMsingle library).")
    col = i - 1  # convert to 0-based
    base_h = hrfs[:, col].astype(float)

    def hrf_callable(t_r, oversampling=1, time_length=library_time_length):
        # Nilearn uses dt = t_r / oversampling and typically np.arange(0, time_length, dt)
        dt = float(t_r) / int(oversampling)
        t_out = np.arange(0.0, float(time_length), dt)  # stop-exclusive (matches nilearn)
        # Interpolate from the library grid to the requested grid
        # Values beyond the library end are set to 0
        h_out = np.interp(t_out, base_time, base_h, left=0.0, right=0.0)
        # Safety: keep peak normalized (library should already be)
        peak = np.max(np.abs(h_out))
        if peak > 0:
            h_out = h_out / peak
        return h_out

    return hrf_callable

def fit_and_run_glm(
    lss_glms: List,
    run_imgs: List,
    events: List,
    confounds: pd.DataFrame,
    hrfs_indices: List,
) -> List[np.ndarray]:

    betas = []
    for lss_glm in lss_glms:
        lss_glm.fit(run_imgs=run_imgs, events=events, confounds = confounds)
        dm = lss_glm.design_matrices_[0]
        # get the beta map and mask it
        beta_map = lss_glm.compute_contrast("probe_hrf_callable", output_type="effect_size")
        beta_map_np = beta_map.get_fdata()
        betas.append(beta_map_np)
        # beta_map_np = fast_apply_mask(target=beta_map_np,mask=union_mask_img.get_fdata())
        # all_betas.append(beta_map_np)

    # hrfs_indices is a numpy array with same shape than beta_map_np with numbers from 0 to 19
    betas = np.array(betas).squeeze(-1)
    idx_expanded = hrfs_indices[np.newaxis, ...]  
    betas = np.take_along_axis(betas, idx_expanded, axis=0)[0]

    return betas, dm

    
    
    