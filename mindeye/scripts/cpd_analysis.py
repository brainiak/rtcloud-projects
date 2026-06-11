"""Cortical Pairmate Distinctiveness (CPD) analysis for sub-005 ses-07.

Computes, for two beta-estimation strategies, how distinctly the brain
represents each MST stimulus relative to its pairmate foil, as a function of
how the trial is temporally modeled:

  1. Varying stimulus-length GLM -- model the probe as a box of length
     L = 1..21 s (causal/cumulative LSS, references = prior trials at 21 s),
     fit per trial, yielding a CPD-vs-modeled-duration curve.
  2. Averaged-BOLD (no GLM) -- average the masked motion-corrected/resampled
     volumes 4-8 s post-onset; pass the raw averaged BOLD straight into MindEye
     (betas are z-scored before the model anyway).

Both feed: z-score -> MindEye ridge+backbone -> predicted CLIP `clip_voxels`
-> utils_mindeye.compute_cpd + retrieval. Diffusion reconstruction is skipped.

The pre-computed motion-corrected/resampled volumes under
`derivatives/motion_corrected_resampled` are reused as-is (no MCFLIRT/FLIRT).

This module's top level is import-cheap on purpose (pure helpers only); the
heavy model/GLM machinery lives inside functions and `main()`.
"""
import glob
import math
import os

import numpy as np
import pandas as pd

# ---- session constants (sub-005 ses-07, task C) --------------------------
TR_LENGTH = 1.5
N_RUNS = 6
N_VOLS_PER_RUN = 239           # resampled vols 0000..0238 per run
TRIALS_PER_RUN = 12
TRUE_STIM_DURATION = 21.0      # real stimulus on-time (s)
DURATIONS = list(range(1, 22))  # varying-length GLM models: L = 1..21 s
AVG_BOLD_WINDOW = (4.0, 8.0)   # post-onset window (s) for the no-GLM strategy
HRF_TAIL = 8.0                 # s of post-stimulus data kept for the causal GLM


# ==========================================================================
# Pure helpers (unit-tested in tests/test_cpd.py)
# ==========================================================================
def _pair_prefix(filename):
    """`pair_9_13_44.png` -> `pair_9_13` (drops the trailing morph suffix)."""
    base = os.path.basename(filename)
    stem = base[:-4] if base.lower().endswith(".png") else os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "pair":
        raise ValueError(f"Not an MST pair filename: {filename}")
    return "_".join(parts[:-1])


def foil_path_for(image_path, available=None):
    """Return the path of an MST image's pairmate foil.

    The foil is the *other* morph that shares the same ``pair_A_B`` prefix
    (e.g. ``pair_9_13_44.png`` -> ``pair_9_13_89.png``). Raises ``ValueError``
    unless exactly one such pairmate exists.

    available : optional list of candidate filenames. If None, the directory of
                ``image_path`` is globbed for ``{prefix}_*.png``.
    """
    directory = os.path.dirname(image_path)
    base = os.path.basename(image_path)
    prefix = _pair_prefix(base)
    if available is None:
        pattern = os.path.join(directory or ".", f"{prefix}_*.png")
        available = [os.path.basename(p) for p in glob.glob(pattern)]
    matches = []
    for f in available:
        try:
            if _pair_prefix(f) == prefix and os.path.basename(f) != base:
                matches.append(os.path.basename(f))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one pairmate for {base} (prefix {prefix}), found {matches}"
        )
    foil_base = matches[0]
    return os.path.join(directory, foil_base) if directory else foil_base


def causal_volume_indices(onset, duration_L, tr_length=TR_LENGTH,
                          n_vols=N_VOLS_PER_RUN, hrf_tail=HRF_TAIL):
    """Contiguous volume indices 0..k available to a *causal/cumulative* GLM.

    Includes every volume from the run start up to the last one acquired at or
    before ``onset + duration_L + hrf_tail`` (so the probe's full HRF response
    is captured), clamped to the run's available volumes. ``onset`` is in
    run-rezeroed seconds (first stimulus onset == 0).
    """
    cutoff = onset + duration_L + hrf_tail
    k = int(math.floor(cutoff / tr_length))
    k = max(0, min(k, n_vols - 1))
    return list(range(0, k + 1))


def avg_bold_volume_indices(onset, tr_length=TR_LENGTH, n_vols=N_VOLS_PER_RUN,
                            lo=AVG_BOLD_WINDOW[0], hi=AVG_BOLD_WINDOW[1]):
    """Volume indices whose acquisition *midpoint* falls in [onset+lo, onset+hi].

    Volume ``i`` is acquired over [i*TR, (i+1)*TR]; its midpoint is
    ``(i+0.5)*TR``. Used by the no-GLM averaged-BOLD strategy (~3 volumes).
    """
    idx = []
    for i in range(n_vols):
        mid = (i + 0.5) * tr_length - onset
        if lo <= mid <= hi:
            idx.append(i)
    return idx


def build_lss_events(events_df, probe_trial_number, probe_duration):
    """Build an LSS events table for one probe trial.

    Onsets are re-zeroed to the run's first onset. Only trials up to and
    including the probe are kept (causal). The probe trial is labelled
    ``"probe"`` with duration ``probe_duration``; all other (reference) trials
    keep their true duration and are labelled ``"reference"``.

    Returns a DataFrame with columns ['onset', 'duration', 'trial_type'].
    """
    df = events_df.copy()
    df["onset"] = df["onset"].astype(float)
    df["duration"] = df["duration"].astype(float)
    first_onset = df["onset"].iloc[0]
    df["onset"] = df["onset"] - first_onset
    df = df[df["trial_number"] <= probe_trial_number].copy()
    df["trial_type"] = np.where(
        df["trial_number"] == probe_trial_number, "probe", "reference"
    )
    df.loc[df["trial_type"] == "probe", "duration"] = float(probe_duration)
    return df[["onset", "duration", "trial_type"]].reset_index(drop=True)
