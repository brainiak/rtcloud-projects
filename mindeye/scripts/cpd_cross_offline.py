#!/usr/bin/env python3
"""Cross-analysis: run the OFFLINE GLMsingle betas through THIS real-time decoder.

Given offline tensors saved under ``cpd_ses-07/verify/offline_data/``:
  - ``sub-005_ses-07_stimdur-03``         images (72,3,224,224)  [stimdur 3s]
  - ``sub-005_ses-07_images_stimdur-21``  images (72,3,224,224)  [stimdur 21s]
  - ``sub-005_ses-07_vox_stimdur-03``     betas  (72,2792)  already z-scored
  - ``sub-005_ses-07_vox_stimdur-21``     betas  (72,2792)  already z-scored

this script:
  1. SANITY-CHECKS the image tensors -- the stimuli must be identical across
     stimdur 03 / 21 (offline) and identical to the real-time pipeline's images
     (same order). A mismatch means trial/foil alignment is off.
  2. CORRELATES the offline betas against this pipeline's cached betas (each GLM
     duration L and the avg-BOLD strategy), per trial, and writes a histogram of
     the spatial-correlation values -- a sanity check that voxel ordering /
     pattern content agree.
  3. CROSS-FEEDS the offline betas through this decoder (``make_predict_fn`` ->
     explicit 2-AFC + CPD + 36-way retrieval), so we can tell whether the
     real-time-vs-offline 2-AFC gap follows the *betas* or the *scoring code*.

Outputs -> ``cpd_ses-07/verify/cross_offline/``.
"""
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

OFFLINE_FILES = {
    "img03": "sub-005_ses-07_stimdur-03",
    "img21": "sub-005_ses-07_images_stimdur-21",
    "vox03": "sub-005_ses-07_vox_stimdur-03",
    "vox21": "sub-005_ses-07_vox_stimdur-21",
}


def _norm(x):
    t = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.normalize(t, dim=-1)


def per_trial_spatial_corr(A, B):
    """Per-trial Pearson r across voxels. A,B: (n_trials, n_vox) -> (n_trials,)."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    num = (Ac * Bc).sum(axis=1)
    den = np.sqrt((Ac ** 2).sum(axis=1) * (Bc ** 2).sum(axis=1)) + 1e-12
    return num / den


def main():
    cfg = C.load_config()
    out_root = os.path.join(cfg["derivatives_path"], "cpd_ses-07")
    off_dir = os.path.join(out_root, "verify", "offline_data")
    cross_dir = os.path.join(out_root, "verify", "cross_offline")
    os.makedirs(cross_dir, exist_ok=True)

    def load(key):
        return torch.load(os.path.join(off_dir, OFFLINE_FILES[key]),
                          map_location="cpu", weights_only=True).float()

    img03, img21 = load("img03"), load("img21")
    vox03, vox21 = load("vox03"), load("vox21")
    n_trials = vox03.shape[0]
    print(f"offline: images {tuple(img03.shape)}  betas {tuple(vox03.shape)}")

    # =====================================================================
    # 1. IMAGE SANITY CHECK
    # =====================================================================
    print("\n[1] image sanity check")
    d_off = float((img03 - img21).abs().max())
    print(f"  offline stimdur-03 vs stimdur-21 images: max|diff| = {d_off:.3e}  "
          f"({'IDENTICAL' if d_off < 1e-5 else 'DIFFER !!'})")

    # real-time images in trial order (from trial_images.csv -> load_image_tensor)
    trials = pd.read_csv(os.path.join(out_root, "trial_images.csv"))
    target_names = trials["image_name"].tolist()
    foil_names = trials["foil_name"].tolist()
    rt_imgs = torch.stack([C.load_image_tensor(cfg, n) for n in target_names])  # (72,3,224,224)
    O = img21.flatten(1).numpy(); R = rt_imgs.flatten(1).numpy()
    d_rt = np.abs(O - R).max(axis=1)                            # per-trial max pixel diff
    img_corr = per_trial_spatial_corr(O, R)                     # per-trial correlation
    n_match = int((img_corr > 0.99).sum())
    print(f"  offline vs real-time images (per-trial): {n_match}/{n_trials} match "
          f"(corr>0.99); mean corr={img_corr.mean():.4f} min={img_corr.min():.4f}; "
          f"max|pixel diff|={d_rt.max():.3e} (sub-pixel resize/antialias delta, not a reorder)")
    if n_match < n_trials:
        bad = np.where(img_corr <= 0.99)[0].tolist()
        print(f"  !! genuinely mismatching trials (possible reorder): {bad}")
    order_ok = n_match == n_trials

    # =====================================================================
    # 2. BETA CORRELATION vs every cached real-time approach
    # =====================================================================
    print("\n[2] beta correlation (per-trial spatial Pearson r)")
    durations = C.DURATIONS
    # real-time betas are raw -> z-score (global) to match the offline z-scoring
    rt_glm = {L: C.zscore_betas(np.load(os.path.join(out_root, f"glm_betas_L{L:02d}.npy")))
              for L in durations}
    rt_avg = C.zscore_betas(np.load(os.path.join(out_root, "avgbold_betas.npy")))

    off = {"03": vox03.numpy(), "21": vox21.numpy()}
    # mean per-trial correlation of each offline set vs each RT approach
    corr_curves = {"03": {}, "21": {}}
    for sd in ("03", "21"):
        for L in durations:
            corr_curves[sd][f"L{L:02d}"] = per_trial_spatial_corr(off[sd], rt_glm[L])
        corr_curves[sd]["avgbold"] = per_trial_spatial_corr(off[sd], rt_avg)

    summ_rows = []
    for sd in ("03", "21"):
        for key, r in corr_curves[sd].items():
            summ_rows.append({"offline": f"stimdur-{sd}", "rt_approach": key,
                              "mean_r": round(float(np.mean(r)), 4),
                              "median_r": round(float(np.median(r)), 4),
                              "min_r": round(float(np.min(r)), 4),
                              "max_r": round(float(np.max(r)), 4)})
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(cross_dir, "beta_correlation_summary.csv"), index=False)
    # matched-stimdur headline
    for sd, Lm in (("03", "L03"), ("21", "L21")):
        rm = corr_curves[sd][Lm]; ra = corr_curves[sd]["avgbold"]
        print(f"  offline-{sd} vs RT-{Lm}: mean r={np.mean(rm):+.3f} "
              f"(median {np.median(rm):+.3f}) | vs RT-avgbold: mean r={np.mean(ra):+.3f}")

    # histogram figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sd, Lm in ((axes[0], "03", "L03"), (axes[1], "21", "L21")):
        bins = np.linspace(-1, 1, 41)
        ax.hist(corr_curves[sd][Lm], bins=bins, alpha=0.6,
                label=f"vs RT-{Lm} (matched)", color="tab:blue")
        ax.hist(corr_curves[sd]["avgbold"], bins=bins, alpha=0.6,
                label="vs RT-avgbold", color="tab:red")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"offline stimdur-{sd} betas vs real-time betas\n"
                     f"(per-trial spatial corr, n={n_trials})")
        ax.set_xlabel("Pearson r"); ax.set_ylabel("# trials"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cross_dir, "beta_correlation_hist.png"), dpi=150)
    plt.close(fig)

    # mean-corr-vs-duration curve (which RT model does each offline set match best)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sd, c in (("03", "tab:green"), ("21", "tab:purple")):
        means = [np.mean(corr_curves[sd][f"L{L:02d}"]) for L in durations]
        ax.plot(durations, means, marker="o", color=c, label=f"offline stimdur-{sd}")
        ax.axhline(np.mean(corr_curves[sd]["avgbold"]), color=c, ls="--", alpha=0.5,
                   label=f"offline-{sd} vs RT-avgbold")
    ax.set_xlabel("real-time GLM modeled duration L (s)")
    ax.set_ylabel("mean per-trial spatial corr")
    ax.set_title("Which real-time GLM duration best matches the offline betas?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(cross_dir, "beta_correlation_vs_duration.png"), dpi=150)
    plt.close(fig)

    # per-trial correlation values (matched) for inspection
    pd.DataFrame({
        "trial": np.arange(1, n_trials + 1),
        "target": [os.path.basename(n) for n in target_names],
        "corr_off03_vs_RTL03": corr_curves["03"]["L03"],
        "corr_off21_vs_RTL21": corr_curves["21"]["L21"],
        "corr_off03_vs_avgbold": corr_curves["03"]["avgbold"],
        "corr_off21_vs_avgbold": corr_curves["21"]["avgbold"],
    }).to_csv(os.path.join(cross_dir, "beta_correlation_per_trial.csv"), index=False)

    # =====================================================================
    # 3. CROSS-FEED: offline betas -> this decoder -> 2-AFC / CPD / retrieval
    # =====================================================================
    print("\n[3] cross-feed: offline betas through THIS decoder")
    if not order_ok:
        print("  WARNING: image order mismatch -> correct/foil mapping may be wrong; "
              "2-AFC below assumes real-time trial order.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C.build_model(cfg, device)
    clip_img_embedder = C.load_clip_embedder(cfg, device)
    predict_fn = C.make_predict_fn(model, device)

    all_names = sorted(set(target_names) | set(foil_names))
    emb = C.embed_unique_images(cfg, clip_img_embedder, all_names, device)
    pool_names = sorted(set(target_names))
    pool_mat = torch.stack([_norm(emb[n]) for n in pool_names])
    pool_idx = {n: i for i, n in enumerate(pool_names)}
    correct_embeds = [emb[n] for n in target_names]
    foil_embeds = [emb[n] for n in foil_names]
    correct_pool_idx = [pool_idx[n] for n in target_names]

    headline = []
    for sd, betas in (("03", vox03), ("21", vox21)):
        rows = []
        for t in range(n_trials):
            # betas are ALREADY z-scored offline -> pass straight in (no re-z-score)
            bt = betas[t].reshape(1, 1, C.NUM_VOXELS).float()
            pred = predict_fn(bt)
            cpd = cpd_from_embeddings(correct=correct_embeds[t], foil=foil_embeds[t], pred=pred)
            p, cE, fE = _norm(pred), _norm(correct_embeds[t]), _norm(foil_embeds[t])
            sim_c, sim_f = float(p @ cE), float(p @ fE)
            afc = 1 if sim_c > sim_f else 0
            sims = (pool_mat @ p).numpy()
            order = np.argsort(-sims)
            rank_of = {int(j): r + 1 for r, j in enumerate(order)}
            rows.append({
                "trial": t + 1, "run": t // 12 + 1, "within_run_trial": t % 12 + 1,
                "target": os.path.basename(target_names[t]),
                "foil": os.path.basename(foil_names[t]),
                "cpd": round(cpd, 6), "twoafc_explicit": afc,
                "twoafc_cpd_sign": 1 if cpd > 0 else 0,
                "agree": bool((1 if cpd > 0 else 0) == afc),
                "sim_correct": round(sim_c, 6), "sim_foil": round(sim_f, 6),
                "correct_rank": rank_of[correct_pool_idx[t]],
                "foil_rank": rank_of[pool_idx[foil_names[t]]],
                "top1_image": pool_names[order[0]],
                "top1_is_correct": int(order[0] == correct_pool_idx[t]),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cross_dir, f"offline_per_trial_stimdur-{sd}.csv"), index=False)
        afc = df["twoafc_explicit"].mean(); top1 = df["top1_is_correct"].mean()
        mrank = df["correct_rank"].median(); nagree = int(df["agree"].sum())
        headline.append((sd, afc, top1, mrank, nagree))
        print(f"  offline stimdur-{sd}: 2AFC={afc:.3f}  top1={top1:.3f}  "
              f"medianRank={mrank:.0f}  agree={nagree}/{n_trials}")

    # =====================================================================
    # Summary comparison vs real-time (causal-z numbers)
    # =====================================================================
    print("\n=== SUMMARY: 2-AFC ===")
    print("  real-time (causal-z, this decoder):  L03=0.736  L21=0.792")
    for sd, afc, top1, mrank, nagree in headline:
        print(f"  offline stimdur-{sd} through this decoder: 2AFC={afc:.3f}")
    print(f"\noutputs -> {cross_dir}")


if __name__ == "__main__":
    main()
