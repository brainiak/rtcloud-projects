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

Run (project venv):
  python scripts/cpd_analysis.py --runs 1 --durations 1,8,21   # smoke
  python scripts/cpd_analysis.py                                # full
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# Make the flat-layout project importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_PROJECT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils_mindeye import cpd_from_embeddings  # noqa: E402

# ---- session constants (sub-005 ses-07, task C) --------------------------
SUB = "sub-005"
SESSION = "ses-07"
TASK = "C"
TR_LENGTH = 1.5
N_RUNS = 6
N_VOLS_PER_RUN = 239           # resampled vols 0000..0238 per run
TRIALS_PER_RUN = 12
NUM_VOXELS = 2792
IMSIZE = 224
TRUE_STIM_DURATION = 21.0      # real stimulus on-time (s)
DURATIONS = list(range(1, 22))  # varying-length GLM models: L = 1..21 s
AVG_BOLD_WINDOW = (4.0, 8.0)   # post-onset window (s) for the no-GLM strategy
HRF_TAIL = 8.0                 # s of post-stimulus data kept for the causal GLM
MODEL_NAME = "sub-005_ses-01_task-C_bs24_MST_rishab_MSTsplit_0_avgrepeats_finalmask"
RELMASK_NAME = "sub-005_ses-01_task-C_relmask.npy"
BOLDREF_NAME = "sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz"  # ses-01 ref reused
STIM_DIR = "all_stimuli/MST_pairs_styleGAN"


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


def build_lss_events(events_df, probe_trial_number, probe_duration,
                     reference_mode="true"):
    """Build an LSS events table for one probe trial.

    Onsets are re-zeroed to the run's first onset. Only trials up to and
    including the probe are kept (causal). The probe trial is labelled
    ``"probe"`` with duration ``probe_duration``; all other (reference) trials
    are labelled ``"reference"``.

    ``reference_mode`` controls the reference trials' modeled duration:
      - ``"true"`` (default): references keep their true (21 s) duration -- an
        *asymmetric* model (probe = L, others = 21 s).
      - ``"matched"``: references are also modeled at ``probe_duration`` -- a
        *symmetric/matched-length* model (probe = L, others = L).

    Returns a DataFrame with columns ['onset', 'duration', 'trial_type'].
    """
    if reference_mode not in ("true", "matched"):
        raise ValueError(
            f"reference_mode must be 'true' or 'matched', got {reference_mode!r}")
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
    if reference_mode == "matched":
        df.loc[df["trial_type"] == "reference", "duration"] = float(probe_duration)
    return df[["onset", "duration", "trial_type"]].reset_index(drop=True)


def sliding_window_onsets(trial_onset, stim_dur=TRUE_STIM_DURATION, box=3.0,
                          step=TR_LENGTH):
    """Absolute onsets of a ``box``-second window stepped by ``step`` across a trial.

    The window starts at ``trial_onset`` and advances in ``step`` increments while
    the whole box stays inside the [trial_onset, trial_onset+stim_dur] stimulus
    period (``offset + box <= stim_dur``). With the defaults (stim_dur=21, box=3,
    step=1.5) this yields 13 windows at offsets 0, 1.5, ..., 18 s.
    """
    onsets = []
    offset = 0.0
    while offset + box <= stim_dur + 1e-9:
        onsets.append(trial_onset + offset)
        offset += step
    return onsets


def zscore_betas(betas):
    """Z-score a (n_trials, n_voxels) beta matrix per voxel, across ALL trials.

    Non-causal (uses the whole session's statistics). Kept for reference/tests;
    the analysis uses ``causal_zscore_betas`` instead.
    """
    betas = np.asarray(betas, dtype=np.float64)
    mean = betas.mean(axis=0, keepdims=True)
    std = betas.std(axis=0, keepdims=True)
    return (betas - mean) / (std + 1e-6)


def causal_zscore_betas(betas):
    """Causally z-score a (n_trials, n_voxels) beta matrix per voxel.

    Trials are assumed to be in temporal order. Row ``t`` is z-scored using the
    per-voxel mean/std computed over trials ``0..t`` *inclusive* only -- the data
    available up to and including that trial -- matching the real-time pipeline's
    running z-score (``mindeye.py`` z_mean/z_std over betas collected so far).
    Row 0 therefore comes out all-zeros (single-sample std == 0).
    """
    betas = np.asarray(betas, dtype=np.float64)
    out = np.empty_like(betas)
    for t in range(betas.shape[0]):
        prefix = betas[: t + 1]
        out[t] = (betas[t] - prefix.mean(axis=0)) / (prefix.std(axis=0) + 1e-6)
    return out


# ==========================================================================
# Aggregation helpers (unit-tested with mocks; no GPU/data required)
# ==========================================================================
def per_trial_cpd(betas, correct_embeds, foil_embeds, predict_fn,
                  num_voxels=NUM_VOXELS, normalize=True):
    """Per-trial CPD for one beta-estimation strategy.

    betas          : (n_trials, num_voxels) beta matrix (z-scored internally).
    correct_embeds : list of CLIP embeddings for each trial's presented image.
    foil_embeds    : list of CLIP embeddings for each trial's pairmate foil.
    predict_fn     : callable betas_tt[(1,1,num_voxels)] -> predicted CLIP embedding.
    Returns np.ndarray (n_trials,) of CPD values.
    """
    z = causal_zscore_betas(betas)
    cpds = []
    for t in range(len(z)):
        betas_tt = torch.from_numpy(np.asarray(z[t], dtype=np.float32)).reshape(
            1, 1, num_voxels
        )
        pred = predict_fn(betas_tt)
        cpds.append(cpd_from_embeddings(correct=correct_embeds[t],
                                        foil=foil_embeds[t],
                                        pred=pred, normalize=normalize))
    return np.asarray(cpds, dtype=np.float64)


def _flat_norm(x):
    if isinstance(x, torch.Tensor):
        v = x.detach().to(torch.float32).reshape(-1).cpu()
    else:
        v = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)
    return F.normalize(v, dim=-1)


def pairmate_2afc(pred, correct, foil):
    """1.0 if predicted embedding is cosine-closer to ``correct`` than ``foil``."""
    p, c, f = _flat_norm(pred), _flat_norm(correct), _flat_norm(foil)
    return 1.0 if float(torch.dot(p, c)) > float(torch.dot(p, f)) else 0.0


def pairmate_2afc_accuracy(predicted, correct_embeds, foil_embeds):
    """Mean pairmate 2-AFC accuracy over trials (chance = 0.5)."""
    accs = [pairmate_2afc(predicted[t], correct_embeds[t], foil_embeds[t])
            for t in range(len(predicted))]
    return float(np.mean(accs)) if accs else float("nan")


def forward_retrieval_accuracy(predicted, correct_pool_idx, pool_embeds):
    """Top-1 forward retrieval accuracy against a fixed image pool.

    predicted        : list of predicted CLIP embeddings (one per trial).
    correct_pool_idx : list of the pool index of each trial's correct image.
    pool_embeds      : list/array of pool image CLIP embeddings.
    Returns mean top-1 accuracy (chance = 1/len(pool)).
    """
    pool = torch.stack([_flat_norm(e) for e in pool_embeds])  # (P, D)
    hits = []
    for t in range(len(predicted)):
        p = _flat_norm(predicted[t])
        sims = pool @ p
        hits.append(1.0 if int(torch.argmax(sims)) == correct_pool_idx[t] else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


# ==========================================================================
# Heavy machinery: config, model, masks, volumes, images (run-time only)
# ==========================================================================
def load_config():
    with open(os.path.join(_PROJECT, "conf", "config.json")) as f:
        return json.load(f)


def build_model(cfg, device):
    """Build MindEyeModule and load the trained checkpoint (strict)."""
    import utils_mindeye
    model = utils_mindeye.MindEyeModule(
        num_voxels=NUM_VOXELS, hidden_dim=1024, seq_len=1,
        clip_emb_dim=1664, clip_seq_dim=256, n_blocks=4,
    )
    # diffusion_prior must exist for strict load, even though we never sample it
    model.build_diffusion_prior(
        clip_emb_dim=1664, clip_seq_dim=256, depth=6,
        dim_head=52, heads=1664 // 52, timesteps=100,
        cond_drop_prob=0.2, image_embed_scale=None,
    )
    ckpt_path = os.path.join(cfg["data_path"], "model", f"{MODEL_NAME}.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device).eval().requires_grad_(False)
    return model


def load_clip_embedder(cfg, device):
    import pickle
    with open(os.path.join(cfg["storage_path"], "clip_img_embedder"), "rb") as f:
        clip_img_embedder = pickle.load(f)
    clip_img_embedder.to(device)
    return clip_img_embedder


def build_union_mask_img(cfg):
    """Replicate notebook cells 13-15: map the 2792-voxel relmask back into the
    19174-voxel nsdgeneral final mask -> a 3D Nifti union mask."""
    import nibabel as nib
    mask_img = nib.load(os.path.join(cfg["data_path"], f"{SUB}_final_mask.nii.gz"))
    union_mask = np.load(os.path.join(cfg["data_path"], RELMASK_NAME))
    mask_data = mask_img.get_fdata().astype(bool)
    true_voxel_indices = np.where(mask_data.ravel())[0]
    selected = true_voxel_indices[union_mask]
    new_flat = np.zeros(mask_data.size, dtype=bool)
    new_flat[selected] = True
    new_data = new_flat.reshape(mask_data.shape)
    union_mask_img = nib.Nifti1Image(new_data.astype(np.uint8), affine=mask_img.affine)
    return union_mask_img


def _fast_apply_mask(target, mask):
    return target[np.where(mask == 1)].T


def load_run_volumes(cfg, run_num):
    """Load all resampled volumes for a run as a float32 4D array (X,Y,Z,T)."""
    from nilearn.image import get_data
    mc_dir = os.path.join(cfg["derivatives_path"], "motion_corrected_resampled")
    vols = []
    for tr in range(N_VOLS_PER_RUN):
        fn = os.path.join(mc_dir, f"{SESSION}_run-{run_num:02d}_{tr:04d}_mc_boldres.nii.gz")
        vols.append(np.asarray(get_data(fn), dtype=np.float32))
    return np.stack(vols, axis=-1)  # (X, Y, Z, T)


def load_run_events(cfg, run_num):
    fn = os.path.join(
        cfg["data_path"], "events",
        f"{SUB}_{SESSION}_task-{TASK}_run-{run_num:02d}_events.tsv",
    )
    return pd.read_csv(fn, sep="\t", header=0)


def load_image_tensor(cfg, image_name, imsize=IMSIZE):
    """Load a stimulus image (path relative to data_path) -> (3, imsize, imsize)."""
    import imageio.v2 as imageio
    from torchvision import transforms
    im = imageio.imread(os.path.join(cfg["data_path"], image_name))
    im = torch.tensor(im / 255.0).permute(2, 0, 1).float()
    # antialias=False to match the canonical stimulus preprocessing the decoder was
    # trained/evaluated with (mindeye.py:281 `transforms.Resize((imsize, imsize))`,
    # which defaults to antialias=False for tensors). Using antialias=True here shifts
    # embeddings at the sub-pixel level and flips ~3/72 near-tie pairmate 2-AFC trials.
    im = transforms.Resize((imsize, imsize), antialias=False)(im.unsqueeze(0))
    return im.squeeze(0)


def embed_unique_images(cfg, clip_img_embedder, image_names, device, imsize=IMSIZE):
    """Embed each unique stimulus image once -> {image_name: clip embedding (cpu float)}."""
    cache = {}
    use_amp = str(device).startswith("cuda")
    for name in sorted(set(image_names)):
        img = load_image_tensor(cfg, name, imsize).reshape(1, 3, imsize, imsize).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    emb = clip_img_embedder(img)
            else:
                emb = clip_img_embedder(img)
        cache[name] = emb.float().cpu()
    return cache


def make_predict_fn(model, device):
    """Return betas_tt -> predicted CLIP `clip_voxels` (ridge+backbone only)."""
    use_amp = str(device).startswith("cuda")

    def predict(betas_tt):
        with torch.no_grad():
            voxel = betas_tt.to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    voxel_ridge = model.ridge(voxel[:, [0]], 0)
                    _, clip_voxels, _ = model.backbone(voxel_ridge)
            else:
                voxel_ridge = model.ridge(voxel[:, [0]], 0)
                _, clip_voxels, _ = model.backbone(voxel_ridge)
        return clip_voxels.float().cpu()

    return predict


# ==========================================================================
# Beta estimation (one beta vector per trial, in the 2792-voxel union space)
# ==========================================================================
def _make_first_level_model(union_mask_img):
    """The single canonical FirstLevelModel config shared by every GLM strategy."""
    from nilearn.glm.first_level import FirstLevelModel
    return FirstLevelModel(
        t_r=TR_LENGTH, slice_time_ref=0, hrf_model="glover", drift_model="cosine",
        drift_order=1, high_pass=0.01, mask_img=union_mask_img, signal_scaling=False,
        smoothing_fwhm=None, noise_model="ar1", n_jobs=1, verbose=0,
        memory_level=1, minimize_memory=True,
    )


def _fit_probe_beta(vols4d, union_mask_img, boldref_nib, events, last_vol):
    """Fit the canonical GLM on volumes 0..last_vol and return the masked 'probe' beta."""
    from nilearn.image import new_img_like
    img = new_img_like(boldref_nib, vols4d[..., : last_vol + 1], copy_header=True)
    glm = _make_first_level_model(union_mask_img)
    glm.fit(run_imgs=img, events=events)
    beta = glm.compute_contrast("probe", output_type="effect_size").get_fdata()
    return _fast_apply_mask(beta, union_mask_img.get_fdata())


def glm_beta_for_trial(vols4d, events_df, union_mask_img, probe_trial_number,
                       probe_duration, boldref_nib, reference_mode="true"):
    """Causal/cumulative LSS beta for one probe trial at a given modeled duration.

    ``reference_mode`` is forwarded to :func:`build_lss_events`: ``"true"`` keeps
    prior reference trials at 21 s (asymmetric); ``"matched"`` models them at the
    same length L as the probe (symmetric/matched-length).
    """
    first_onset = float(events_df["onset"].astype(float).iloc[0])
    probe_onset = float(events_df["onset"].astype(float).iloc[probe_trial_number]) - first_onset
    vol_idx = causal_volume_indices(probe_onset, probe_duration)
    events = build_lss_events(events_df, probe_trial_number, probe_duration,
                              reference_mode=reference_mode)
    return _fit_probe_beta(vols4d, union_mask_img, boldref_nib, events, vol_idx[-1])


def glm_matched_beta_for_trial(vols4d, events_df, union_mask_img, probe_trial_number,
                               probe_duration, boldref_nib):
    """Matched-length variant: probe = L AND prior references = L (symmetric)."""
    return glm_beta_for_trial(vols4d, events_df, union_mask_img, probe_trial_number,
                              probe_duration, boldref_nib, reference_mode="matched")


def sliding_beta_for_trial(vols4d, events_df, union_mask_img, probe_trial_number,
                           boldref_nib, box=3.0):
    """Per-window betas for one trial: a ``box``-second probe stepped 1 TR at a time.

    For each window position (``sliding_window_onsets``) a causal GLM is fit with
    the probe = a ``box``-second boxcar at that position, and all *strictly prior*
    trials kept as 21 s references. Volumes are truncated to
    ``window_onset + box + hrf_tail``. Returns (n_windows, NUM_VOXELS)."""
    onset_col = events_df["onset"].astype(float)
    first_onset = float(onset_col.iloc[0])
    trial_onset = float(onset_col.iloc[probe_trial_number]) - first_onset

    # prior trials (strictly before the probe) at their true duration -> references
    refs = events_df[events_df["trial_number"] < probe_trial_number].copy()
    ref_onsets = (refs["onset"].astype(float) - first_onset).tolist()
    ref_durs = refs["duration"].astype(float).tolist()

    betas = []
    for win_onset in sliding_window_onsets(trial_onset, box=box):
        rows = {
            "onset": ref_onsets + [win_onset],
            "duration": ref_durs + [float(box)],
            "trial_type": ["reference"] * len(ref_onsets) + ["probe"],
        }
        events = pd.DataFrame(rows)
        vol_idx = causal_volume_indices(win_onset, box)
        betas.append(_fit_probe_beta(vols4d, union_mask_img, boldref_nib,
                                     events, vol_idx[-1]))
    return np.asarray(betas)


def avgbold_beta_for_trial(vols4d, events_df, union_mask_img, trial_number):
    """No-GLM averaged-BOLD beta: mean of masked volumes 4-8 s post-onset."""
    first_onset = float(events_df["onset"].astype(float).iloc[0])
    onset = float(events_df["onset"].astype(float).iloc[trial_number]) - first_onset
    vol_idx = avg_bold_volume_indices(onset)
    avg = vols4d[..., vol_idx].mean(axis=-1)
    return _fast_apply_mask(avg, union_mask_img.get_fdata())


# ==========================================================================
# Orchestration
# ==========================================================================
def compute_betas(cfg, union_mask_img, boldref_nib, runs, durations, strategies):
    """Compute all betas. Returns dict:
        betas['glm'][L]         -> (n_trials, NUM_VOXELS)   asymmetric (refs=21s)
        betas['glm_matched'][L] -> (n_trials, NUM_VOXELS)   matched (refs=L)
        betas['avgbold']        -> (n_trials, NUM_VOXELS)
        betas['sliding']        -> (n_trials, n_windows, NUM_VOXELS)
    plus the parallel list trial_image_names (correct image per trial), kept in
    a single consistent trial order across all strategies.
    """
    glm_betas = {L: [] for L in durations} if "glm" in strategies else {}
    matched_betas = {L: [] for L in durations} if "glm_matched" in strategies else {}
    avg_betas = [] if "avgbold" in strategies else None
    sliding_betas = [] if "sliding" in strategies else None
    trial_image_names = []

    for run_num in runs:
        events_df = load_run_events(cfg, run_num)
        vols4d = load_run_volumes(cfg, run_num)
        n_trials = len(events_df)
        for trial in range(n_trials):
            trial_image_names.append(str(events_df["image_name"].iloc[trial]))
            if "avgbold" in strategies:
                avg_betas.append(
                    avgbold_beta_for_trial(vols4d, events_df, union_mask_img, trial)
                )
            if "glm" in strategies:
                for L in durations:
                    glm_betas[L].append(
                        glm_beta_for_trial(vols4d, events_df, union_mask_img,
                                           trial, float(L), boldref_nib)
                    )
            if "glm_matched" in strategies:
                for L in durations:
                    matched_betas[L].append(
                        glm_matched_beta_for_trial(vols4d, events_df, union_mask_img,
                                                   trial, float(L), boldref_nib)
                    )
            if "sliding" in strategies:
                sliding_betas.append(
                    sliding_beta_for_trial(vols4d, events_df, union_mask_img,
                                           trial, boldref_nib)
                )
        print(f"  run {run_num}: {n_trials} trials done", flush=True)

    out = {}
    if "glm" in strategies:
        out["glm"] = {L: np.asarray(v) for L, v in glm_betas.items()}
    if "glm_matched" in strategies:
        out["glm_matched"] = {L: np.asarray(v) for L, v in matched_betas.items()}
    if "avgbold" in strategies:
        out["avgbold"] = np.asarray(avg_betas)
    if "sliding" in strategies:
        out["sliding"] = np.asarray(sliding_betas)  # (n_trials, n_windows, NUM_VOXELS)
    return out, trial_image_names


def analyze(cfg, betas, trial_image_names, clip_img_embedder, predict_fn, device,
            durations, strategies):
    """Turn betas into per-trial CPD + retrieval summaries for each strategy."""
    emb_cache = embed_unique_images(cfg, clip_img_embedder, trial_image_names, device)
    available = [os.path.basename(p) for p in
                 glob.glob(os.path.join(cfg["data_path"], STIM_DIR, "*.png"))]
    foil_names = [foil_path_for(n, available=available) for n in trial_image_names]
    # embed any foils not already in the presented-image set
    missing = [fn for fn in set(foil_names) if fn not in emb_cache]
    if missing:
        emb_cache.update(embed_unique_images(cfg, clip_img_embedder, missing, device))

    correct_embeds = [emb_cache[n] for n in trial_image_names]
    foil_embeds = [emb_cache[fn] for fn in foil_names]

    # pool for forward retrieval = unique correct images
    pool_names = sorted(set(trial_image_names))
    pool_embeds = [emb_cache[n] for n in pool_names]
    pool_idx = {n: i for i, n in enumerate(pool_names)}
    correct_pool_idx = [pool_idx[n] for n in trial_image_names]

    results = {"trial_image_names": trial_image_names, "foil_names": foil_names}

    def score(b):
        """Causal z-score -> predict -> CPD/2-AFC/retrieval for a (n_trials, NUM_VOXELS) matrix."""
        z = causal_zscore_betas(b)
        preds = [predict_fn(torch.from_numpy(np.asarray(z[t], dtype=np.float32))
                            .reshape(1, 1, NUM_VOXELS)) for t in range(len(z))]
        cpd = np.asarray([cpd_from_embeddings(correct=correct_embeds[t],
                                              foil=foil_embeds[t], pred=preds[t])
                          for t in range(len(preds))])
        return {
            "cpd": cpd,
            "twoafc": pairmate_2afc_accuracy(preds, correct_embeds, foil_embeds),
            "retrieval": forward_retrieval_accuracy(preds, correct_pool_idx, pool_embeds),
        }

    for key, label in (("glm", "GLM"), ("glm_matched", "GLMm")):
        if key in strategies:
            results[key] = {}
            for L in durations:
                r = score(betas[key][L])
                results[key][L] = r
                print(f"  {label} L={L:>2}s  meanCPD={r['cpd'].mean():+.3f}  "
                      f"2AFC={r['twoafc']:.2f}  ret={r['retrieval']:.2f}", flush=True)

    if "avgbold" in strategies:
        r = score(betas["avgbold"])
        results["avgbold"] = r
        print(f"  AVGBOLD  meanCPD={r['cpd'].mean():+.3f}  "
              f"2AFC={r['twoafc']:.2f}  ret={r['retrieval']:.2f}", flush=True)

    # 'sliding' has no CPD summary here -- its deliverable is reliability
    # (computed from the cached betas by cpd_reliability.py).
    return results


def save_results(out_dir, betas, results, durations, strategies):
    os.makedirs(out_dir, exist_ok=True)
    # both per-duration GLM variants share the same on-disk layout, distinguished
    # by a filename prefix ("glm" asymmetric, "glm_matched" matched-length).
    for key in ("glm", "glm_matched"):
        if key in strategies:
            np.save(os.path.join(out_dir, f"{key}_cpd_per_trial.npy"),
                    np.stack([results[key][L]["cpd"] for L in durations]))  # (n_dur, n_trials)
            np.save(os.path.join(out_dir, f"{key}_durations.npy"), np.asarray(durations))
            np.save(os.path.join(out_dir, f"{key}_2afc.npy"),
                    np.asarray([results[key][L]["twoafc"] for L in durations]))
            np.save(os.path.join(out_dir, f"{key}_retrieval.npy"),
                    np.asarray([results[key][L]["retrieval"] for L in durations]))
            for L in durations:
                np.save(os.path.join(out_dir, f"{key}_betas_L{L:02d}.npy"), betas[key][L])
    if "avgbold" in strategies:
        np.save(os.path.join(out_dir, "avgbold_cpd_per_trial.npy"), results["avgbold"]["cpd"])
        np.save(os.path.join(out_dir, "avgbold_betas.npy"), betas["avgbold"])
        with open(os.path.join(out_dir, "avgbold_summary.json"), "w") as f:
            json.dump({"twoafc": results["avgbold"]["twoafc"],
                       "retrieval": results["avgbold"]["retrieval"]}, f, indent=2)
    if "sliding" in strategies:
        np.save(os.path.join(out_dir, "sliding_betas.npy"), betas["sliding"])
    pd.DataFrame({"image_name": results["trial_image_names"],
                  "foil_name": results["foil_names"]}).to_csv(
        os.path.join(out_dir, "trial_images.csv"), index=False)


def _offline_metric(out_dir, stimdur, col):
    """Mean of a per-trial column from the offline GLMsingle benchmark, or None.

    Reads cross_offline/offline_per_trial_stimdur-{NN}.csv (the offline benchmark
    exports). col is e.g. "twoafc_explicit" or "top1_is_correct".
    """
    import csv
    path = os.path.join(out_dir, "verify", "cross_offline",
                        f"offline_per_trial_stimdur-{stimdur:02d}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    return sum(float(r[col]) for r in rows) / len(rows)


def _retrieval_pool_size(out_dir, results, durations):
    """Unique-image retrieval pool size (chance = 1/pool). Reads trial_images.csv,
    falling back to n_trials//2 (each image shown twice)."""
    import csv
    ti = os.path.join(out_dir, "trial_images.csv")
    if os.path.exists(ti):
        return len({r["image_name"] for r in csv.DictReader(open(ti))})
    any_key = next(k for k in ("glm", "glm_matched") if k in results)
    return len(results[any_key][durations[0]]["cpd"]) // 2


def _plot_metric_vs_duration(out_dir, results, durations, strategies, plt, spec):
    """One metric (2-AFC or retrieval) vs modeled length, both GLM variants overlaid.

    asym + matched real-time GLM curves and the avg-BOLD/chance references stay
    in the legend; the offline GLMsingle benchmark is drawn as black dash-dot
    lines annotated with floating label boxes (one per stimdur).
    """
    import matplotlib.transforms as mtransforms

    glm_series = [("glm", "asym GLM", "tab:blue", "o"),
                  ("glm_matched", "matched GLM", "tab:orange", "s")]
    glm_series = [g for g in glm_series if g[0] in strategies]

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, klabel, color, marker in glm_series:
        ax.plot(durations, [results[key][L][spec["metric"]] for L in durations],
                marker=marker, color=color, label=f"real-time {klabel}")
    if "avgbold" in strategies:
        ax.axhline(results["avgbold"][spec["metric"]], color="dimgray", ls="--",
                   alpha=0.7, label="avg-BOLD")
    ax.axhline(spec["chance_y"], color="gray", lw=0.8, ls=":", label=spec["chance_label"])
    ax.legend(loc="best")

    # offline benchmark: black dash-dot lines with floating label boxes (no legend).
    # lower value labeled below its line, higher above, to avoid overlap.
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    offs = [(d, _offline_metric(out_dir, d, spec["off_col"])) for d in (3, 21)]
    offs = sorted([(d, v) for d, v in offs if v is not None], key=lambda x: x[1])
    for rank, (d, v) in enumerate(offs):
        ax.axhline(v, color="black", ls="-.", alpha=0.8)
        va = "top" if (rank == 0 and len(offs) > 1) else "bottom"
        ax.text(0.02, v, f"offline {spec['off_label']} (stimdur {d}s) = {v:.3f}",
                transform=trans, va=va, ha="left", fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9))

    ax.set_xlabel("modeled stimulus length (s)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(f"{SUB} {SESSION}: {spec['title']} vs modeled stimulus length")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, spec["fname"]), dpi=150)
    plt.close(fig)


def plot_results(out_dir, results, durations, strategies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "glm" in strategies:
        cpd = np.stack([results["glm"][L]["cpd"] for L in durations])  # (n_dur, n_trials)
        mean = cpd.mean(axis=1)
        sem = cpd.std(axis=1) / np.sqrt(cpd.shape[1])

        # main: CPD vs modeled duration (averaged over trials)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(durations, mean, yerr=sem, marker="o", label="varying-length GLM")
        if "avgbold" in strategies:
            ab = results["avgbold"]["cpd"]
            ax.axhline(ab.mean(), color="tab:red", ls="--",
                       label=f"avg-BOLD 4-8s ({ab.mean():+.2f})")
            ax.fill_between(durations, ab.mean() - ab.std() / np.sqrt(len(ab)),
                            ab.mean() + ab.std() / np.sqrt(len(ab)),
                            color="tab:red", alpha=0.15)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("modeled stimulus length (s)")
        ax.set_ylabel("CPD  (+1 correct / -1 foil)")
        ax.set_title(f"{SUB} {SESSION}: CPD vs modeled stimulus length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cpd_vs_duration.png"), dpi=150)
        plt.close(fig)

        # per-trial heatmap (trials x durations); data-driven symmetric scale
        absmax = max(float(np.nanpercentile(np.abs(cpd), 98)), 1e-6)
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.imshow(cpd.T, aspect="auto", cmap="RdBu_r", vmin=-absmax, vmax=absmax,
                       extent=[durations[0], durations[-1], cpd.shape[1], 0])
        ax.set_xlabel("modeled stimulus length (s)")
        ax.set_ylabel("trial (image presentation)")
        ax.set_title(f"{SUB} {SESSION}: per-trial CPD")
        fig.colorbar(im, ax=ax, label="CPD")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cpd_per_trial_heatmap.png"), dpi=150)
        plt.close(fig)

        # 2-AFC + retrieval vs duration: one plot each, both GLM variants overlaid
        pool_size = _retrieval_pool_size(out_dir, results, durations)
        for spec in (
            dict(metric="twoafc", off_col="twoafc_explicit", off_label="2-AFC",
                 ylabel="pairmate 2-AFC accuracy", title="pairmate 2-AFC",
                 chance_y=0.5, chance_label="chance (0.5)",
                 fname="twoafc_vs_duration.png"),
            dict(metric="retrieval", off_col="top1_is_correct", off_label="retrieval",
                 ylabel="top-1 retrieval accuracy", title="top-1 retrieval",
                 chance_y=1.0 / pool_size,
                 chance_label=f"chance (1/{pool_size}={1.0/pool_size:.3f})",
                 fname="retrieval_vs_duration.png"),
        ):
            _plot_metric_vs_duration(out_dir, results, durations, strategies, plt, spec)


def load_cached_results(out_dir, strategies):
    """Rebuild the `results` dict from cached .npy/.json (for --replot)."""
    results, durations = {}, []
    for key in ("glm", "glm_matched"):
        if key in strategies:
            durations = np.load(os.path.join(out_dir, f"{key}_durations.npy")).tolist()
            cpd = np.load(os.path.join(out_dir, f"{key}_cpd_per_trial.npy"))  # (n_dur, n_trials)
            twoafc = np.load(os.path.join(out_dir, f"{key}_2afc.npy"))
            ret = np.load(os.path.join(out_dir, f"{key}_retrieval.npy"))
            results[key] = {L: {"cpd": cpd[i], "twoafc": float(twoafc[i]),
                                "retrieval": float(ret[i])}
                            for i, L in enumerate(durations)}
    if "avgbold" in strategies:
        with open(os.path.join(out_dir, "avgbold_summary.json")) as f:
            summ = json.load(f)
        results["avgbold"] = {
            "cpd": np.load(os.path.join(out_dir, "avgbold_cpd_per_trial.npy")),
            "twoafc": summ["twoafc"], "retrieval": summ["retrieval"]}
    return results, durations


def load_cached_betas(out_dir, durations, strategies):
    """Reload betas + trial image names saved by a previous full run.

    Lets us re-run the downstream (z-score -> predict -> CPD/2-AFC) without
    refitting any GLMs -- e.g. after changing the z-scoring scheme.
    """
    betas = {}
    for key in ("glm", "glm_matched"):
        if key in strategies:
            betas[key] = {L: np.load(os.path.join(out_dir, f"{key}_betas_L{L:02d}.npy"))
                          for L in durations}
    if "avgbold" in strategies:
        betas["avgbold"] = np.load(os.path.join(out_dir, "avgbold_betas.npy"))
    if "sliding" in strategies:
        betas["sliding"] = np.load(os.path.join(out_dir, "sliding_betas.npy"))
    trial_image_names = pd.read_csv(
        os.path.join(out_dir, "trial_images.csv"))["image_name"].tolist()
    return betas, trial_image_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="all",
                    help="comma-separated run numbers (1-6) or 'all'")
    ap.add_argument("--durations", default="all",
                    help="comma-separated durations (1-21) or 'all'")
    ap.add_argument("--strategies", default="glm,avgbold")
    ap.add_argument("--out", default=None, help="output dir (default derivatives/cpd_ses-07)")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate plots from cached .npy without recomputing betas")
    ap.add_argument("--reanalyze", action="store_true",
                    help="reload cached betas and re-run CPD/2-AFC + save + plot "
                         "(no GLM refit); use after changing z-scoring/analysis")
    args = ap.parse_args()

    cfg = load_config()
    runs = list(range(1, N_RUNS + 1)) if args.runs == "all" else \
        [int(x) for x in args.runs.split(",")]
    durations = DURATIONS if args.durations == "all" else \
        [int(x) for x in args.durations.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    out_dir = args.out or os.path.join(cfg["derivatives_path"], "cpd_ses-07")

    if args.replot:
        results, durations = load_cached_results(out_dir, strategies)
        plot_results(out_dir, results, durations, strategies)
        print(f"replotted -> {out_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.reanalyze:
        print(f"reanalyze (causal z-score) device={device} durations={durations} "
              f"strategies={strategies}")
        betas, trial_image_names = load_cached_betas(out_dir, durations, strategies)
        model = build_model(cfg, device)
        clip_img_embedder = load_clip_embedder(cfg, device)
        predict_fn = make_predict_fn(model, device)
        results = analyze(cfg, betas, trial_image_names, clip_img_embedder,
                          predict_fn, device, durations, strategies)
        save_results(out_dir, betas, results, durations, strategies)
        plot_results(out_dir, results, durations, strategies)
        print(f"reanalyzed -> {out_dir}")
        return

    print(f"device={device} runs={runs} durations={durations} strategies={strategies}")

    import nibabel as nib
    boldref_nib = nib.load(os.path.join(cfg["data_path"], BOLDREF_NAME))
    union_mask_img = build_union_mask_img(cfg)
    model = build_model(cfg, device)
    clip_img_embedder = load_clip_embedder(cfg, device)
    predict_fn = make_predict_fn(model, device)

    print("computing betas...")
    betas, trial_image_names = compute_betas(
        cfg, union_mask_img, boldref_nib, runs, durations, strategies)
    print(f"computed betas for {len(trial_image_names)} trials")

    print("analyzing (CPD + retrieval)...")
    results = analyze(cfg, betas, trial_image_names, clip_img_embedder,
                      predict_fn, device, durations, strategies)

    save_results(out_dir, betas, results, durations, strategies)
    plot_results(out_dir, results, durations, strategies)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
