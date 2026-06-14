#!/usr/bin/env python3
"""Score the sliding-window 3s GLM betas with CPD / 2-AFC / retrieval.

The main `cpd_analysis.analyze` path computes these metrics for the varying-length
GLM and avg-BOLD but deliberately SKIPS the sliding strategy (its headline there is
reliability). This script fills that gap: it loads the cached
`sliding_betas.npy` (n_trials, n_windows, NUM_VOXELS) and runs the SAME scoring
pipeline the GLM betas get -- causal z-score -> decoder -> CPD/2-AFC/retrieval -- for

  1. each individual window position (metric vs window center time), and
  2. averages over window subsets (all windows, early/peak/late bands, each half,
     and cumulative-from-onset averages of the first k windows).

A window i is a 3s box at offset i*1.5s, i.e. covering [i*1.5, i*1.5+3]s post-onset
(center (i+1)*1.5s); with 21s stimuli there are 13 windows (centers 1.5..19.5s).

Reference: the varying-length GLM at L=21 (cached `glm_2afc.npy` etc).
Outputs -> cpd_ses-07/sliding_eval/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "models")))

import cpd_analysis as C
from utils_mindeye import cpd_from_embeddings


def main():
    cfg = C.load_config()
    out_root = os.path.join(cfg["derivatives_path"], "cpd_ses-07")
    out_dir = os.path.join(out_root, "sliding_eval")
    os.makedirs(out_dir, exist_ok=True)

    sliding = np.load(os.path.join(out_root, "sliding_betas.npy"))  # (n_trials, n_win, n_vox)
    n_trials, n_win, n_vox = sliding.shape
    centers = [(i + 1) * C.TR_LENGTH for i in range(n_win)]  # window center time (s)
    print(f"sliding betas {sliding.shape}; {n_win} windows, centers {centers[0]}..{centers[-1]}s")

    trials = pd.read_csv(os.path.join(out_root, "trial_images.csv"))
    target_names = trials["image_name"].tolist()
    foil_names = trials["foil_name"].tolist()

    # ---- embeddings (identical setup to cpd_analysis.analyze) -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C.build_model(cfg, device)
    clip = C.load_clip_embedder(cfg, device)
    predict_fn = C.make_predict_fn(model, device)

    all_names = sorted(set(target_names) | set(foil_names))
    emb = C.embed_unique_images(cfg, clip, all_names, device)
    correct_embeds = [emb[n] for n in target_names]
    foil_embeds = [emb[n] for n in foil_names]
    pool_names = sorted(set(target_names))
    pool_embeds = [emb[n] for n in pool_names]
    pool_idx = {n: i for i, n in enumerate(pool_names)}
    correct_pool_idx = [pool_idx[n] for n in target_names]

    def score(b2d):
        """(n_trials, n_vox) raw betas -> dict(meanCPD, twoafc, retrieval)."""
        z = C.causal_zscore_betas(b2d)
        preds = [predict_fn(torch.from_numpy(np.asarray(z[t], dtype=np.float32))
                            .reshape(1, 1, C.NUM_VOXELS)) for t in range(len(z))]
        cpd = np.asarray([cpd_from_embeddings(correct=correct_embeds[t],
                                              foil=foil_embeds[t], pred=preds[t])
                          for t in range(len(preds))])
        return {"meanCPD": float(cpd.mean()),
                "twoafc": C.pairmate_2afc_accuracy(preds, correct_embeds, foil_embeds),
                "retrieval": C.forward_retrieval_accuracy(preds, correct_pool_idx, pool_embeds)}

    # ---- 1. per-window --------------------------------------------------------
    print("\n[1] per-window")
    per_window = []
    for w in range(n_win):
        r = score(sliding[:, w, :])
        per_window.append(r)
        print(f"  win {w:2d} (center {centers[w]:4.1f}s): "
              f"meanCPD={r['meanCPD']:+.4f}  2AFC={r['twoafc']:.3f}  ret={r['retrieval']:.3f}")

    # ---- 2. averaged subsets --------------------------------------------------
    # contiguous index bands chosen by window-center time
    subsets = {
        "all (1.5-19.5s)":   list(range(n_win)),
        "early (1.5-4.5s)":  [0, 1, 2],
        "peak (4.5-9s)":     [2, 3, 4, 5],
        "mid (6-15s)":       list(range(3, 10)),
        "late (15-19.5s)":   [9, 10, 11, 12],
        "first_half":        list(range(0, n_win // 2)),
        "second_half":       list(range(n_win // 2, n_win)),
    }
    print("\n[2] averaged over window subsets (mean betas -> score)")
    subset_res = {}
    for name, sel in subsets.items():
        r = score(sliding[:, sel, :].mean(axis=1))
        subset_res[name] = {**r, "windows": sel}
        print(f"  {name:18s} (n={len(sel):2d}): meanCPD={r['meanCPD']:+.4f}  "
              f"2AFC={r['twoafc']:.3f}  ret={r['retrieval']:.3f}")

    # cumulative-from-onset: average of the first k windows
    print("\n[3] cumulative average of first k windows")
    cumulative = []
    for k in range(1, n_win + 1):
        r = score(sliding[:, :k, :].mean(axis=1))
        cumulative.append(r)
        print(f"  first {k:2d} win (..{centers[k-1]:4.1f}s): meanCPD={r['meanCPD']:+.4f}  "
              f"2AFC={r['twoafc']:.3f}  ret={r['retrieval']:.3f}")

    # ---- reference: varying-length GLM at L=21 --------------------------------
    ref = {}
    try:
        dur = np.load(os.path.join(out_root, "glm_durations.npy")).astype(int).tolist()
        i21 = dur.index(21)
        ref = {"twoafc": float(np.load(os.path.join(out_root, "glm_2afc.npy"))[i21]),
               "retrieval": float(np.load(os.path.join(out_root, "glm_retrieval.npy"))[i21]),
               "meanCPD": float(np.load(os.path.join(out_root, "glm_cpd_per_trial.npy"))[i21].mean())}
        print(f"\nreference GLM L=21: meanCPD={ref['meanCPD']:+.4f}  "
              f"2AFC={ref['twoafc']:.3f}  ret={ref['retrieval']:.3f}")
    except (FileNotFoundError, ValueError):
        print("\n(no cached GLM L=21 reference found)")

    # ---- save -----------------------------------------------------------------
    summary = {
        "window_centers_s": centers,
        "per_window": per_window,
        "subsets": subset_res,
        "cumulative_first_k": cumulative,
        "glm_L21_reference": ref,
    }
    with open(os.path.join(out_dir, "sliding_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _plot(out_dir, centers, per_window, subset_res, cumulative, ref)
    print(f"\noutputs -> {out_dir}")


def _plot(out_dir, centers, per_window, subset_res, cumulative, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("twoafc", "pairmate 2-AFC", 0.5),
               ("retrieval", "top-1 retrieval", None),
               ("meanCPD", "mean CPD", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ks = list(range(1, len(cumulative) + 1))
    for ax, (m, label, chance) in zip(axes, metrics):
        ax.plot(centers, [w[m] for w in per_window], marker="o",
                color="tab:blue", label="per-window")
        ax.plot(centers, [c[m] for c in cumulative], marker="^",
                color="tab:green", label="cumulative (first k)")
        ax.axhline(subset_res["all (1.5-19.5s)"][m], color="tab:orange", ls="--",
                   alpha=0.8, label="avg all windows")
        if ref:
            ax.axhline(ref[m], color="black", ls="-.", alpha=0.8, label="GLM L=21")
        if chance is not None:
            ax.axhline(chance, color="gray", lw=0.8, ls=":", label="chance")
        ax.set_xlabel("window center time (s post-onset)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle(f"{C.SUB} {C.SESSION}: sliding 3s-window GLM -- CPD / 2-AFC / retrieval")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sliding_eval.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
