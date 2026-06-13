#!/usr/bin/env python3
"""Verification / intermediate-output dump for the CPD analysis (sub-005 ses-07).

Reuses the *cached* betas produced by ``cpd_analysis.py`` (no GLM refit) to:

  1. Re-run the MindEye ridge+backbone predictions per trial.
  2. Compute the pairmate **2-AFC explicitly** from dot products
     (``pred.correct > pred.foil``) -- NOT derived from the CPD sign -- and
     independently compare it against ``CPD > 0`` so any internal bug surfaces.
  3. Compute the full **forward-retrieval ranking over all unique images**
     (repeats excluded -> 36 unique morphs) for every trial, recording the rank
     of the correct image and of the foil.

All intermediates are written under ``<out>/verify/`` so nothing is recomputed
implicitly:

  - ``per_trial_L{LL}.csv`` / ``per_trial_avgbold.csv`` -- one row per trial with
    cpd, twoafc_explicit, twoafc_cpd_sign, agree, sim_correct, sim_foil,
    correct_rank, foil_rank, top1_image.
  - ``retrieval_sorted_L{LL}.npz`` / ``retrieval_sorted_avgbold.npz`` -- pool of
    36 names plus, per trial, the descending-sorted pool indices and cosine sims
    (the full 36-image ranking).
  - ``twoafc_explicit.npy`` -- (n_cond, n_trials) explicit 2-AFC matrix.
  - ``afc_cpd_mismatches.csv`` -- every (condition, trial) where explicit 2-AFC
    disagrees with CPD>0 (empty == clean).

Run:  python cpd_verify.py            # all cached glm durations + avgbold
      python cpd_verify.py --durations 19,21
"""
import argparse
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


def _norm(x):
    """Flatten to (D,) float and L2-normalize -- matches utils_mindeye._flat_norm."""
    t = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.normalize(t, dim=-1)


def explicit_2afc(pred, correct, foil):
    """1.0 iff predicted embedding is cosine-closer to correct than foil.

    Independent of cpd_from_embeddings -- computed straight from dot products."""
    p, c, f = _norm(pred), _norm(correct), _norm(foil)
    return 1.0 if float(p @ c) > float(p @ f) else 0.0, float(p @ c), float(p @ f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--durations", default="all",
                    help="comma-separated GLM durations to verify, or 'all' cached")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = C.load_config()
    out_dir = args.out or os.path.join(cfg["derivatives_path"], "cpd_ses-07")
    verify_dir = os.path.join(out_dir, "verify")
    os.makedirs(verify_dir, exist_ok=True)

    # ---- trial target/foil names (exactly as the analysis used them) ----------
    trials = pd.read_csv(os.path.join(out_dir, "trial_images.csv"))
    target_names = trials["image_name"].tolist()
    foil_names = trials["foil_name"].tolist()
    n_trials = len(target_names)
    print(f"{n_trials} trials")

    # ---- which conditions to verify -------------------------------------------
    if args.durations == "all":
        cached = sorted(int(f[len("glm_betas_L"):-len(".npy")])
                        for f in os.listdir(out_dir)
                        if f.startswith("glm_betas_L") and f.endswith(".npy"))
    else:
        cached = [int(x) for x in args.durations.split(",")]
    have_avgbold = os.path.exists(os.path.join(out_dir, "avgbold_betas.npy"))
    print(f"glm durations: {cached}   avgbold: {have_avgbold}")

    # ---- model + embedder + predict fn ----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; loading model + clip embedder ...")
    model = C.build_model(cfg, device)
    clip_img_embedder = C.load_clip_embedder(cfg, device)
    predict_fn = C.make_predict_fn(model, device)

    # ---- embed every UNIQUE image once (repeats excluded) ---------------------
    pool_names = sorted(set(target_names))           # unique morphs presented = 36
    all_needed = sorted(set(target_names) | set(foil_names))
    emb = C.embed_unique_images(cfg, clip_img_embedder, all_needed, device)
    print(f"unique images embedded: {len(emb)}   retrieval pool: {len(pool_names)}")
    assert all(n in emb for n in pool_names)
    pool_idx = {n: i for i, n in enumerate(pool_names)}
    pool_mat = torch.stack([_norm(emb[n]) for n in pool_names])   # (P, D)

    correct_embeds = [emb[n] for n in target_names]
    foil_embeds = [emb[n] for n in foil_names]
    correct_pool_idx = [pool_idx[n] for n in target_names]

    # ---- run each condition ----------------------------------------------------
    conditions = [("glm", L) for L in cached] + ([("avgbold", None)] if have_avgbold else [])
    afc_matrix = np.zeros((len(conditions), n_trials), dtype=np.float64)
    cond_labels = []
    all_mismatches = []

    for ci, (strat, L) in enumerate(conditions):
        if strat == "glm":
            label = f"L{L:02d}"
            betas = np.load(os.path.join(out_dir, f"glm_betas_L{L:02d}.npy"))
        else:
            label = "avgbold"
            betas = np.load(os.path.join(out_dir, "avgbold_betas.npy"))
        cond_labels.append(label)

        z = C.causal_zscore_betas(betas)             # identical to analyze()
        rows = []
        sorted_idx = np.zeros((n_trials, len(pool_names)), dtype=np.int32)
        sorted_sim = np.zeros((n_trials, len(pool_names)), dtype=np.float32)

        for t in range(n_trials):
            bt = torch.from_numpy(np.asarray(z[t], dtype=np.float32)).reshape(1, 1, C.NUM_VOXELS)
            pred = predict_fn(bt)                    # (1,256,1664) cpu float

            # CPD (the pipeline's own function) and EXPLICIT 2-AFC (independent)
            cpd = cpd_from_embeddings(correct=correct_embeds[t], foil=foil_embeds[t], pred=pred)
            afc, sim_c, sim_f = explicit_2afc(pred, correct_embeds[t], foil_embeds[t])
            afc_cpd_sign = 1.0 if cpd > 0 else 0.0
            agree = (afc == afc_cpd_sign)
            if not agree:
                all_mismatches.append({"condition": label, "trial": t + 1,
                                       "cpd": cpd, "twoafc_explicit": afc,
                                       "sim_correct": sim_c, "sim_foil": sim_f})
            afc_matrix[ci, t] = afc

            # full retrieval ranking over the 36-image pool (descending sim)
            p = _norm(pred)
            sims = (pool_mat @ p).numpy()
            order = np.argsort(-sims)                # best first
            sorted_idx[t] = order
            sorted_sim[t] = sims[order]
            rank_of = {int(j): r + 1 for r, j in enumerate(order)}  # 1-based
            correct_rank = rank_of[correct_pool_idx[t]]
            foil_rank = rank_of[pool_idx[foil_names[t]]]

            rows.append({
                "trial": t + 1,
                "run": t // 12 + 1,
                "within_run_trial": t % 12 + 1,
                "target": os.path.basename(target_names[t]),
                "foil": os.path.basename(foil_names[t]),
                "cpd": round(cpd, 6),
                "twoafc_explicit": int(afc),
                "twoafc_cpd_sign": int(afc_cpd_sign),
                "agree": bool(agree),
                "sim_correct": round(sim_c, 6),
                "sim_foil": round(sim_f, 6),
                "correct_rank": correct_rank,
                "foil_rank": foil_rank,
                "top1_image": pool_names[order[0]],
                "top1_is_correct": int(order[0] == correct_pool_idx[t]),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(verify_dir, f"per_trial_{label}.csv"), index=False)
        np.savez_compressed(
            os.path.join(verify_dir, f"retrieval_sorted_{label}.npz"),
            pool_names=np.array(pool_names),
            sorted_pool_idx=sorted_idx,          # (n_trials, 36)
            sorted_cosine=sorted_sim,            # (n_trials, 36)
            correct_pool_idx=np.array(correct_pool_idx),
            foil_pool_idx=np.array([pool_idx[f] for f in foil_names]),
        )
        n_agree = int(df["agree"].sum())
        print(f"  {label}: 2AFC={df['twoafc_explicit'].mean():.3f}  "
              f"top1={df['top1_is_correct'].mean():.3f}  "
              f"medianCorrectRank={df['correct_rank'].median():.0f}  "
              f"agree(explicit==cpd>0)={n_agree}/{n_trials}")

    np.save(os.path.join(verify_dir, "twoafc_explicit.npy"), afc_matrix)
    with open(os.path.join(verify_dir, "conditions.txt"), "w") as f:
        f.write("\n".join(cond_labels) + "\n")

    mm = pd.DataFrame(all_mismatches)
    mm.to_csv(os.path.join(verify_dir, "afc_cpd_mismatches.csv"), index=False)
    print()
    if len(mm) == 0:
        print("CLEAN: explicit 2-AFC == (CPD>0) for EVERY trial in EVERY condition.")
    else:
        print(f"!!! {len(mm)} (condition,trial) MISMATCHES between explicit 2-AFC and CPD>0:")
        print(mm.to_string(index=False))
    print(f"intermediates -> {verify_dir}")


if __name__ == "__main__":
    main()
