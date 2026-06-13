"""Pairmate 2-AFC evaluation for MST stimuli.

This is the *minimally modified* version of the original ``evaluate_mst_pairs``.
The body of the scoring loop is byte-for-byte the same; the only changes are:

  1. ``image_idx`` (and the other former globals) are now explicit arguments,
     so the caller states the index-space contract at the call site instead of
     it being an implicit global.
  2. A validation preamble (``assert_pairmate_inputs``) runs first and refuses
     to score if ``mst_pairs``/``image_idx``/``vox``/``images`` don't line up.

The bug this guards against: ``vox[image_idx[pair]]`` silently reads the wrong
(or unreachable) rows when ``image_idx`` doesn't match the index space of
``vox``/``images``.

Index-space contract (works for BOTH input modes):
  - ``mst_pairs`` holds indices into ``image_idx`` (pairs-space).
  - ``image_idx`` maps pairs-space -> rows of ``vox``/``images`` (vox-space).
  - Pass the ``image_idx`` that matches your ``vox``:
      * single-trial / per-presentation ``vox`` (1 row per presentation):
            image_idx = np.arange(len(vox))          # identity
      * averaged / per-unique-image ``vox`` (1 row per unique morph):
            image_idx = presentation -> unique-id map  # the original map
"""
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


def batchwise_cosine_similarity(Z, B):
    """Cosine similarity between rows of Z and rows of B (matches utils_mindeye)."""
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    return (Z @ B) / (Z_norm @ B_norm + 1e-8)


def _pair_prefix(name):
    """`.../pair_9_13_44.png` -> `pair_9_13` (the pairmate group)."""
    return "_".join(str(name).split("/")[-1].split("_")[:3])


def assert_pairmate_inputs(mst_pairs, image_idx, vox, images, vox_image_names=None):
    """Validate that ``mst_pairs`` + ``image_idx`` resolve to a clean pairmate 2-AFC.

    Mode-general: works whether ``vox``/``images`` are per-presentation
    (single-trial) or per-unique-image (averaged). Raises ``AssertionError`` on
    any index-space mismatch -- the class of bug where ``vox[image_idx[pair]]``
    reads the wrong / unreachable rows.
    """
    pairs = np.asarray(mst_pairs)
    idx = np.asarray(image_idx)
    n_vox = len(vox)

    assert len(vox) == len(images), \
        f"vox ({len(vox)}) and images ({len(images)}) must be aligned (same rows)"
    assert pairs.ndim == 2 and pairs.shape[1] == 2, \
        f"mst_pairs must be (n_pairs, 2), got shape {pairs.shape}"
    assert pairs.min() >= 0 and pairs.max() < len(idx), \
        f"mst_pairs values must index image_idx (len {len(idx)}); got [{pairs.min()}, {pairs.max()}]"
    assert (pairs[:, 0] != pairs[:, 1]).all(), \
        "self-paired row found (target index == foil index)"

    rows = idx[pairs.reshape(-1)]
    assert rows.min() >= 0 and rows.max() < n_vox, \
        f"image_idx[mst_pairs] out of range for vox (len {n_vox}); got [{rows.min()}, {rows.max()}]"
    # (b) every vox row is reached -> no dead rows / index-space mismatch
    assert set(rows.tolist()) == set(range(n_vox)), \
        ("image_idx[mst_pairs] does not cover every vox row -- index-space mismatch "
         "(e.g. presentation-ordered vox with a unique-id image_idx)")
    # (c) every vox row reached the SAME number of times (1x single-trial, kx averaged)
    counts = np.bincount(rows, minlength=n_vox)
    assert len(set(counts.tolist())) == 1, \
        f"vox rows reached a non-uniform number of times: {sorted(set(counts.tolist()))}"

    # (d) universal safety net: resolved target/foil are genuine pairmates
    if vox_image_names is not None:
        names = np.asarray(vox_image_names)
        assert len(names) == n_vox, \
            f"vox_image_names ({len(names)}) must align with vox ({n_vox})"
        for a, b in pairs:
            na, nb = names[idx[a]], names[idx[b]]
            assert _pair_prefix(na) == _pair_prefix(nb), \
                f"resolved target/foil are not pairmates: {na} vs {nb}"
            assert na != nb, f"resolved target == foil (same morph): {na}"


def evaluate_mst_pairs(mst_pairs, *, vox, images, image_idx, model,
                       clip_img_embedder, device, data_type=torch.float16,
                       vox_image_names=None, check_inputs=True):
    """Pairmate 2-AFC accuracy.

    For each pairmate (A, B): score +1 if the brain decode of A's voxel is
    cosine-closer to image A than image B, and +1 if B's voxel is closer to B
    than A. Returns score / total over all comparisons.
    """
    if check_inputs:
        assert_pairmate_inputs(mst_pairs, image_idx, vox, images, vox_image_names)

    score = 0
    total = 0
    autocast = (torch.cuda.amp.autocast(dtype=data_type)
                if torch.device(device).type == "cuda" else nullcontext())
    with torch.no_grad(), autocast:
        for pair in mst_pairs:
            voxel = vox[image_idx[pair[0]]].to(device)[None]
            voxel = torch.Tensor(voxel).unsqueeze(1).to(device)

            imageA = images[image_idx[pair[0]]].to(device)[None]
            imageB = images[image_idx[pair[1]]].to(device)[None]

            clip_targetA = clip_img_embedder(imageA.float())
            clip_targetB = clip_img_embedder(imageB.float())

            voxel_ridge = model.ridge(voxel, 0)
            backbone, clip_voxels, _ = model.backbone(voxel_ridge)

            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
            clip_targetA_norm = nn.functional.normalize(clip_targetA.flatten(1), dim=-1)
            clip_targetB_norm = nn.functional.normalize(clip_targetB.flatten(1), dim=-1)

            if (batchwise_cosine_similarity(clip_voxels_norm, clip_targetA_norm)
                    > batchwise_cosine_similarity(clip_voxels_norm, clip_targetB_norm)):
                score += 1
            total += 1

            voxel = vox[image_idx[pair[1]]].to(device)[None]
            voxel = torch.Tensor(voxel).unsqueeze(1).to(device)

            voxel_ridge = model.ridge(voxel, 0)
            backbone, clip_voxels, _ = model.backbone(voxel_ridge)
            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)

            if (batchwise_cosine_similarity(clip_voxels_norm, clip_targetB_norm)
                    > batchwise_cosine_similarity(clip_voxels_norm, clip_targetA_norm)):
                score += 1
            total += 1

    return score / total
