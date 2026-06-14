"""Reliability comparison of beta-estimation approaches -- sub-005 ses-07.

ses-07 presents 36 unique morphs, each shown exactly twice (in different runs).
"Reliability" of a beta-estimation approach = the Pearson correlation between the
two repeats' *raw* masked beta patterns (over the 2792 mask voxels), averaged
across the 36 images. Raw betas are used (not the decoder's causal z-scoring):
Pearson is mean/scale invariant, so this isolates estimation reliability.

This module operates purely on the cached `.npy` betas written by
`cpd_analysis.py` (no GLM refits), computes reliability for every approach
(asymmetric GLM @L, matched-length GLM @L, averaged-BOLD, sliding-window 3 s),
plus a cross-approach beta-correlation matrix, and writes arrays + plots to
`<out_dir>/reliability/`.

Run:  python cpd_reliability.py
"""
import os
import re
import glob
import json

import numpy as np
import pandas as pd

from cpd_analysis import load_config, sliding_window_onsets, SUB, SESSION


# ==========================================================================
# Pure metric helpers (unit-tested in tests/test_cpd.py)
# ==========================================================================
def pair_repeats(trial_image_names):
    """Map a trial->image list to the (idx_a, idx_b) index pairs of each image.

    Every image must appear exactly twice (the ses-07 split-half structure).
    Returns a list of (lower_index, higher_index) tuples sorted by lower index.
    Raises ValueError if any image does not appear exactly twice.
    """
    groups = {}
    for i, name in enumerate(trial_image_names):
        groups.setdefault(name, []).append(i)
    bad = {n: len(idx) for n, idx in groups.items() if len(idx) != 2}
    if bad:
        raise ValueError(f"expected exactly 2 repeats per image; offenders: {bad}")
    pairs = [(idx[0], idx[1]) for idx in groups.values()]
    return sorted(pairs)


def _pearson(x, y):
    """Pearson correlation of two 1-D vectors (nan if either is constant)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def repeat_reliability(betas, trial_image_names, return_per_image=False):
    """Mean across-repeat Pearson correlation for a (n_trials, n_vox) beta matrix.

    Correlates each image's two repeats over the voxel axis, then averages over
    images. With ``return_per_image`` also returns the per-image correlation array.
    """
    betas = np.asarray(betas, dtype=np.float64)
    pairs = pair_repeats(trial_image_names)
    per_image = np.array([_pearson(betas[i], betas[j]) for i, j in pairs])
    mean = float(np.nanmean(per_image))
    return (mean, per_image) if return_per_image else mean


def cross_approach_correlation(betas_a, betas_b):
    """Mean per-trial Pearson correlation (over voxels) between two approaches.

    Both inputs are (n_trials, n_vox) raw beta matrices in the same trial order.
    """
    a = np.asarray(betas_a, dtype=np.float64)
    b = np.asarray(betas_b, dtype=np.float64)
    per_trial = np.array([_pearson(a[t], b[t]) for t in range(len(a))])
    return float(np.nanmean(per_trial))


def within_presentation_reliability(sliding_betas):
    """Pseudo-repeat reliability: mean pairwise window correlation *within* a trial.

    ``sliding_betas`` is (n_trials, n_windows, n_vox). For each trial we average
    the Pearson correlations over all distinct window pairs, then average across
    trials. Measures within-presentation temporal stability of the 3 s windows.
    """
    sb = np.asarray(sliding_betas, dtype=np.float64)
    n_trials, n_win, _ = sb.shape
    trial_means = []
    for t in range(n_trials):
        rs = [_pearson(sb[t, a], sb[t, b])
              for a in range(n_win) for b in range(a + 1, n_win)]
        trial_means.append(np.nanmean(rs))
    return float(np.nanmean(trial_means))


# ==========================================================================
# Cached-beta loading
# ==========================================================================
def _load_per_duration(out_dir, key):
    """Load {L: (n_trials, n_vox)} for a per-duration GLM strategy, or {} if absent."""
    out = {}
    for path in sorted(glob.glob(os.path.join(out_dir, f"{key}_betas_L*.npy"))):
        m = re.search(rf"{re.escape(key)}_betas_L(\d+)\.npy$", os.path.basename(path))
        if m:
            out[int(m.group(1))] = np.load(path)
    return out


# ==========================================================================
# Orchestration
# ==========================================================================
def compute_all(out_dir):
    """Compute every reliability summary from cached betas. Returns a dict."""
    trial_image_names = pd.read_csv(
        os.path.join(out_dir, "trial_images.csv"))["image_name"].tolist()

    res = {"trial_image_names": trial_image_names}

    # per-duration GLM variants -------------------------------------------
    glm = _load_per_duration(out_dir, "glm")
    matched = _load_per_duration(out_dir, "glm_matched")
    res["glm_durations"] = sorted(glm)
    res["matched_durations"] = sorted(matched)
    res["glm_reliability"] = np.array(
        [repeat_reliability(glm[L], trial_image_names) for L in sorted(glm)])
    res["matched_reliability"] = np.array(
        [repeat_reliability(matched[L], trial_image_names) for L in sorted(matched)])

    # asym-vs-matched cross-correlation per shared L ----------------------
    shared_L = sorted(set(glm) & set(matched))
    res["asym_vs_matched_L"] = shared_L
    res["asym_vs_matched_corr"] = np.array(
        [cross_approach_correlation(glm[L], matched[L]) for L in shared_L])

    # averaged-BOLD -------------------------------------------------------
    avg_path = os.path.join(out_dir, "avgbold_betas.npy")
    if os.path.exists(avg_path):
        avg = np.load(avg_path)
        res["avgbold_reliability"] = repeat_reliability(avg, trial_image_names)
    else:
        avg = None

    # sliding window ------------------------------------------------------
    sliding_path = os.path.join(out_dir, "sliding_betas.npy")
    if os.path.exists(sliding_path):
        sliding = np.load(sliding_path)  # (n_trials, n_windows, n_vox)
        n_win = sliding.shape[1]
        res["sliding_window_times"] = np.array(sliding_window_onsets(0.0))[:n_win]
        res["sliding_per_window_reliability"] = np.array(
            [repeat_reliability(sliding[:, w, :], trial_image_names)
             for w in range(n_win)])
        res["sliding_avg_reliability"] = repeat_reliability(
            sliding.mean(axis=1), trial_image_names)
        res["sliding_pseudo_reliability"] = within_presentation_reliability(sliding)
    else:
        sliding = None

    # cross-approach matrix at a representative L (best asym reliability) --
    approaches = {}
    if len(glm):
        bestL = sorted(glm)[int(np.nanargmax(res["glm_reliability"]))]
        res["cross_repr_L"] = bestL
        approaches[f"glm@{bestL}"] = glm[bestL]
        if bestL in matched:
            approaches[f"matched@{bestL}"] = matched[bestL]
    if avg is not None:
        approaches["avgbold"] = avg
    if sliding is not None:
        approaches["sliding_avg"] = sliding.mean(axis=1)
    labels = list(approaches)
    mat = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = cross_approach_correlation(approaches[a], approaches[b])
    res["cross_labels"] = labels
    res["cross_matrix"] = mat

    return res


def _ranked_summary(res):
    """List of (label, reliability) for every single-number approach, best first."""
    rows = []
    for L, r in zip(res["glm_durations"], res["glm_reliability"]):
        rows.append((f"glm(asym) L={L}", r))
    for L, r in zip(res["matched_durations"], res["matched_reliability"]):
        rows.append((f"glm(matched) L={L}", r))
    if "avgbold_reliability" in res:
        rows.append(("avgbold 4-8s", res["avgbold_reliability"]))
    if "sliding_avg_reliability" in res:
        rows.append(("sliding 3s (avg windows)", res["sliding_avg_reliability"]))
        rows.append(("sliding 3s (pseudo-repeat)", res["sliding_pseudo_reliability"]))
    return sorted(rows, key=lambda kv: (np.isnan(kv[1]), -kv[1]))


def save_reliability(out_dir, res):
    rel_dir = os.path.join(out_dir, "reliability")
    os.makedirs(rel_dir, exist_ok=True)
    np.savez(
        os.path.join(rel_dir, "reliability.npz"),
        glm_durations=np.array(res["glm_durations"]),
        glm_reliability=res["glm_reliability"],
        matched_durations=np.array(res["matched_durations"]),
        matched_reliability=res["matched_reliability"],
        asym_vs_matched_L=np.array(res["asym_vs_matched_L"]),
        asym_vs_matched_corr=res["asym_vs_matched_corr"],
        sliding_window_times=res.get("sliding_window_times", np.array([])),
        sliding_per_window_reliability=res.get("sliding_per_window_reliability", np.array([])),
        cross_matrix=res["cross_matrix"],
    )
    summary = {
        "avgbold_reliability": res.get("avgbold_reliability"),
        "sliding_avg_reliability": res.get("sliding_avg_reliability"),
        "sliding_pseudo_reliability": res.get("sliding_pseudo_reliability"),
        "cross_repr_L": res.get("cross_repr_L"),
        "cross_labels": res.get("cross_labels"),
        "ranking": [[lbl, None if np.isnan(r) else round(float(r), 4)]
                    for lbl, r in _ranked_summary(res)],
    }
    with open(os.path.join(rel_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return rel_dir


def plot_reliability(rel_dir, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. reliability vs modeled length (asym vs matched), with avgbold baseline
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(res["glm_durations"]):
        ax.plot(res["glm_durations"], res["glm_reliability"], marker="o",
                label="GLM asymmetric (refs=21s)")
    if len(res["matched_durations"]):
        ax.plot(res["matched_durations"], res["matched_reliability"], marker="s",
                label="GLM matched (refs=L)")
    if "avgbold_reliability" in res:
        ax.axhline(res["avgbold_reliability"], color="tab:red", ls="--",
                   label=f"avg-BOLD 4-8s ({res['avgbold_reliability']:.3f})")
    if "sliding_avg_reliability" in res:
        ax.axhline(res["sliding_avg_reliability"], color="tab:green", ls=":",
                   label=f"sliding avg ({res['sliding_avg_reliability']:.3f})")
    ax.set_xlabel("modeled stimulus length L (s)")
    ax.set_ylabel("repeat reliability (Pearson r)")
    ax.set_title(f"{SUB} {SESSION}: reliability vs modeled length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(rel_dir, "reliability_vs_length.png"), dpi=150)
    plt.close(fig)

    # 2. sliding per-window reliability vs window time
    if "sliding_per_window_reliability" in res:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(res["sliding_window_times"], res["sliding_per_window_reliability"],
                marker="o", label="per-window (3s box)")
        ax.axhline(res["sliding_avg_reliability"], color="tab:green", ls="--",
                   label=f"averaged windows ({res['sliding_avg_reliability']:.3f})")
        ax.axhline(res["sliding_pseudo_reliability"], color="tab:purple", ls=":",
                   label=f"pseudo-repeat ({res['sliding_pseudo_reliability']:.3f})")
        ax.set_xlabel("window onset (s after trial onset)")
        ax.set_ylabel("repeat reliability (Pearson r)")
        ax.set_title(f"{SUB} {SESSION}: sliding 3s-window reliability vs time")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(rel_dir, "reliability_vs_window_time.png"), dpi=150)
        plt.close(fig)

    # 3. approach comparison bar (best of each)
    ranking = _ranked_summary(res)
    # collapse the per-L GLM families to their best L for a readable bar chart
    best = {}
    for lbl, r in ranking:
        fam = lbl.split(" L=")[0]
        if fam not in best or (not np.isnan(r) and r > best[fam][1]):
            best[fam] = (lbl, r)
    items = sorted(best.values(), key=lambda kv: (np.isnan(kv[1]), -kv[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([lbl for lbl, _ in items], [r for _, r in items], color="tab:blue")
    ax.set_ylabel("repeat reliability (Pearson r)")
    ax.set_title(f"{SUB} {SESSION}: best reliability per approach")
    ax.tick_params(axis="x", rotation=30)
    for i, (_, r) in enumerate(items):
        if not np.isnan(r):
            ax.text(i, r, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(rel_dir, "approach_comparison.png"), dpi=150)
    plt.close(fig)

    # 4. cross-approach correlation heatmap
    labels, mat = res["cross_labels"], res["cross_matrix"]
    if labels:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{SUB} {SESSION}: cross-approach beta correlation")
        fig.colorbar(im, ax=ax, label="mean per-trial r")
        fig.tight_layout()
        fig.savefig(os.path.join(rel_dir, "cross_approach_corr.png"), dpi=150)
        plt.close(fig)


def main():
    cfg = load_config()
    out_dir = os.path.join(cfg["derivatives_path"], "cpd_ses-07")
    res = compute_all(out_dir)
    rel_dir = save_reliability(out_dir, res)
    plot_reliability(rel_dir, res)

    print(f"\nreliability summary ({SUB} {SESSION}) -- highest repeat correlation first:")
    for lbl, r in _ranked_summary(res):
        print(f"  {lbl:<28} r = {r:.4f}")
    print(f"\nsaved arrays + 4 plots -> {rel_dir}")


if __name__ == "__main__":
    main()
